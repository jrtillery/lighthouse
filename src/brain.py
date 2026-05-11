"""Brain module — manages Gemma 4 E2B conversations, modes, and session state."""
import litert_lm
import re
import time
import config
from modes import MODES, EngagementTracker


class LighthouseBrain:
    """Manages the LLM engine, conversation, learning modes, and session state."""

    def __init__(self):
        self.engine = None
        self.conversation = None
        self.history = []
        self.current_topic = None
        self.turn_count = 0

        # Mode management
        self.active_mode = None        # Current mode key (str) or None
        self.mode_turn = 0             # Turns within current mode
        self.engagement = EngagementTracker()
        self._system_prefix = None     # Set by _send_system_prompt(); None = system role supported

    def start(self):
        """Load the model and initialize a conversation."""
        print("[BRAIN] Loading Gemma 4 E2B via LiteRT-LM...")
        start = time.time()
        # TODO: pass max_tokens=config.MAX_RESPONSE_TOKENS once LiteRT-LM exposes that
        # parameter; currently only post-hoc sentence trimming enforces response length.
        self.engine = litert_lm.Engine(config.GEMMA_MODEL)
        self.conversation = self.engine.create_conversation()
        elapsed = time.time() - start
        print(f"[BRAIN] Model loaded in {elapsed:.1f}s")
        self._send_system_prompt()

    def _send_system_prompt(self):
        """Send the base system prompt."""
        try:
            self.conversation.send_message({
                "role": "system",
                "content": config.SYSTEM_PROMPT,
            })
            self._system_prefix = None
            print("[BRAIN] System prompt loaded (system role)")
        except Exception:
            self._system_prefix = config.SYSTEM_PROMPT + "\n\n"
            print("[BRAIN] System role not supported, will prepend to first message")

    def enter_mode(self, mode_key):
        """Activate a learning mode.

        Returns the mode's greeting text (to be spoken by the orchestrator).
        """
        if mode_key not in MODES:
            mode_key = "explore"

        mode = MODES[mode_key]
        self.active_mode = mode_key
        self.mode_turn = 0
        self.engagement.reset()

        print(f"[BRAIN] Entering mode: {mode['name']}")

        # Inject mode-specific system context into the conversation
        mode_context = mode["system_context"]
        try:
            self.conversation.send_message(
                f"[MODE ACTIVATED: {mode['name'].upper()}]\n\n{mode_context}"
            )
        except Exception as e:
            print(f"[BRAIN] Warning: couldn't inject mode context: {e}")

        return mode.get("greeting")

    def exit_mode(self):
        """Deactivate the current mode, return to free explore."""
        if self.active_mode:
            print(f"[BRAIN] Exiting mode: {MODES[self.active_mode]['name']}")
        self.active_mode = None
        self.mode_turn = 0
        self.engagement.reset()

    @property
    def mode_is_complete(self):
        """Check if the current mode has reached its natural end."""
        if self.active_mode is None:
            return False
        max_turns = MODES[self.active_mode].get("max_turns", 99)
        return self.mode_turn >= max_turns

    def think(self, user_text, image_path=None):
        """Send user input to Gemma and get a response.

        Args:
            user_text: What the child said (transcribed)
            image_path: Optional path to captured image

        Returns:
            Response text from Gemma
        """
        self.turn_count += 1
        self.mode_turn += 1
        start = time.time()

        # Track engagement
        self.engagement.record_turn(user_text)

        # Build the message
        # NOTE: Vision encoder is not supported on CPU/Pi 5 via LiteRT-LM
        # (TF_LITE_VISION_ENCODER skipped, max_num_images: 0). Sending image
        # tokens causes a segfault. Fallback: describe the image in text so
        # Gemma can still respond contextually in Show & Tell mode.
        prompt = self._build_prompt(user_text, image_path)
        message = prompt
        if image_path:
            print(f"[BRAIN] Thinking (text-only fallback, mode={self.active_mode})...")
        else:
            print(f"[BRAIN] Thinking (mode={self.active_mode})...")

        try:
            response = self.conversation.send_message(message)
        except Exception as e:
            print(f"[BRAIN] Error: {e}")
            response = "Hmm, I got a little confused. Could you say that again?"

        elapsed = time.time() - start
        # API returns: {'role': 'assistant', 'content': [{'type': 'text', 'text': '...'}]}
        try:
            response_text = response['content'][0]['text'].strip()
        except (KeyError, IndexError, TypeError):
            response_text = str(response).strip()

        # Trim to 3 sentences — matches the spoken 3-sentence cap
        sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
        if len(sentences) > 3:
            response_text = ' '.join(sentences[:3])
            if response_text and response_text[-1] not in '.!?':
                response_text += '.'

        print(f"[BRAIN] Response ({elapsed:.1f}s): \"{response_text[:80]}...\"")
        self._update_state(user_text, response_text)

        return response_text

    def _build_prompt(self, user_text, image_path=None):
        """Build the full prompt with mode context and engagement awareness."""
        parts = []

        # System prefix fallback (first turn only)
        if self._system_prefix and self.turn_count == 1:
            parts.append(self._system_prefix)

        # Mode context reminder (keep Gemma on track)
        if self.active_mode and self.active_mode != "explore":
            mode = MODES[self.active_mode]
            parts.append(
                f"[You are in {mode['name']} mode, turn {self.mode_turn} of {mode['max_turns']}. "
                f"Stay in character for this mode.]"
            )

            # Signal when mode is about to end
            if self.mode_turn >= mode["max_turns"] - 1:
                parts.append("[This is the LAST turn of this mode. Wrap up with a celebration!]")

        # Topic context
        if self.current_topic:
            parts.append(f"[Topic: {self.current_topic}]")

        # The child's input
        if image_path:
            # Vision not available on this hardware — ask child to describe
            parts.append(
                f"The child is holding something up to show you and said: \"{user_text}\"\n"
                "You can't quite see it clearly. Ask them one excited question to get them to "
                "describe it — its color, shape, texture, or what it does. React with curiosity!"
            )
        else:
            parts.append(f"The child says: \"{user_text}\"")

        # Engagement intervention
        if self.engagement.is_disengaged and self.active_mode != "challenge":
            parts.append(
                "[The child seems to be losing interest. "
                "Make your response EXTRA short and exciting. "
                "Suggest something active they can DO right now, "
                "or offer a choice between two fun options.]"
            )

        return '\n'.join(parts)

    def _update_state(self, user_text, response_text):
        """Update session state after a turn."""
        self.history.append({"role": "user", "summary": user_text[:100]})
        self.history.append({"role": "assistant", "summary": response_text[:100]})

        if len(self.history) > config.CONVERSATION_HISTORY_LIMIT * 2:
            self.history = self.history[-config.CONVERSATION_HISTORY_LIMIT * 2:]

        # Topic detection
        if self.current_topic is None:
            text_lower = user_text.lower()
            for kw in ["learn about", "tell me about", "what is", "what are",
                        "i like", "i want to", "let's talk about", "know about"]:
                if kw in text_lower:
                    idx = text_lower.index(kw) + len(kw)
                    topic = user_text[idx:].strip().rstrip('?.,!')[:50]
                    if topic:
                        self.current_topic = topic
                        print(f"[BRAIN] Topic detected: {self.current_topic}")
                    break

    def get_greeting(self):
        """Generate a startup greeting."""
        try:
            response = self.conversation.send_message(
                "A child just walked up to you. Say a warm hello (1 sentence), "
                "then offer them a choice: 'Want to go on a quest, play a game, "
                "or tell me about something cool?' Keep it to 2 sentences total."
            )
            try:
                text = response['content'][0]['text'].strip()
            except (KeyError, IndexError, TypeError):
                text = str(response).strip()
            self._update_state("[child approached]", text)
            return text
        except Exception:
            return (
                "Hey there, explorer! "
                "Want to go on a quest, play a game, or tell me about something cool?"
            )

    def reset_session(self):
        """Reset for a new learning session."""
        self.history.clear()
        self.current_topic = None
        self.turn_count = 0
        self.exit_mode()
        if self.engine:
            try:
                self.conversation = self.engine.create_conversation()
                self._send_system_prompt()
            except Exception as e:
                print(f"[BRAIN] Error resetting conversation: {e}")
        print("[BRAIN] Session reset")

    def shutdown(self):
        """Clean up resources."""
        if self.conversation:
            try:
                self.conversation.__exit__(None, None, None)
            except Exception:
                pass
        if self.engine:
            try:
                self.engine.__exit__(None, None, None)
            except Exception:
                pass
        print("[BRAIN] Shutdown complete")