#!/usr/bin/env python3
"""
Lighthouse — Screen-free AI learning companion for children.

Main orchestrator: listen → detect mode → [see] → think → speak

Usage:
    python lighthouse.py              # Normal mode
    python lighthouse.py --no-camera  # Skip camera (for testing without webcam)
    python lighthouse.py --debug      # Verbose logging
"""
import sys
import time
import signal
import random
import argparse
import threading
import config
import stt
import tts
import camera
import sounds
from brain import LighthouseBrain
from modes import detect_mode, get_random_mode_suggestion, MODES


class Lighthouse:
    """Main orchestrator for the Lighthouse learning companion."""

    def __init__(self, use_camera=True, debug=False):
        self.use_camera = use_camera
        self.debug = debug
        self.running = False
        self.brain = LighthouseBrain()
        self.mic_index = None
        self.turns_since_mode_end = 0  # Track idle turns after a mode ends

    def start(self):
        """Initialize all subsystems and begin the main loop."""
        print("\n" + "=" * 50)
        print("  LIGHTHOUSE — Starting up...")
        print("=" * 50 + "\n")

        sounds.init_sounds()
        sounds.play('startup')
        time.sleep(0.5)

        self.mic_index = stt.find_usb_mic()

        if self.use_camera:
            camera.get_camera()

        tts.speak("Hold on, I'm waking up my brain. This takes a moment.")
        self.brain.start()

        # Pre-render all filler phrases so they play instantly (no render delay)
        tts.prerender_fillers([
            "Ooh, let me think of your next mission...",
            "Great job, explorer!",
            "The adventure continues...",
            "Ooh, and then...",
            "Oh wow, what a twist!",
            "The story gets even better...",
            "Let me come up with a good one...",
            "Ready for this?",
            "Oh, let me see...",
            "That's a really good question!",
            "I love that you're asking why!",
            "Hmm, let me think...",
            "That's interesting...",
        ])

        greeting = self.brain.get_greeting()
        tts.speak(greeting)

        self.running = True
        print("\n[LIGHTHOUSE] Ready! Listening for speech...\n")
        self._main_loop()

    def _main_loop(self):
        """Core interaction loop with mode management."""
        while self.running:
            try:
                # === LISTEN ===
                sounds.play('listening')
                user_text, stt_time = stt.listen_and_transcribe(self.mic_index)

                if user_text is None or not user_text.strip():
                    continue

                text_lower = user_text.lower().strip()

                # === COMMANDS ===
                if self._handle_commands(text_lower):
                    continue

                # === MODE DETECTION ===
                requested_mode = detect_mode(text_lower)

                if requested_mode and requested_mode != self.brain.active_mode:
                    # Child requested a specific mode
                    self._switch_mode(requested_mode)

                    # For show_and_tell, immediately capture after entering mode
                    if requested_mode == "show_and_tell" and self.use_camera:
                        sounds.play('camera')
                        image_path = camera.capture()
                        if image_path:
                            response = self.brain.think(user_text, image_path)
                            tts.speak(response)
                            continue

                    # For other modes, the greeting already set the stage
                    continue

                elif self.brain.active_mode is None and requested_mode is None:
                    # No mode active and none requested — auto-enter explore
                    self.brain.enter_mode("explore")

                # === SEE (camera trigger within any mode) ===
                image_path = None
                if self.use_camera and self._wants_camera(text_lower):
                    # If not already in show_and_tell, enter it
                    if self.brain.active_mode != "show_and_tell":
                        self._switch_mode("show_and_tell")

                    sounds.play('camera')
                    tts.speak("Let me take a look!")
                    image_path = camera.capture()
                    if image_path is None:
                        tts.speak("Hmm, I couldn't see anything. Can you hold it up closer?")
                        continue

                # === THINK (parallel with filler) ===
                # Start Gemma inference immediately in background so it runs
                # concurrently while the filler phrase plays — cuts perceived
                # latency by the filler's full duration (~1-2s).
                response_holder = [None]
                def _think():
                    response_holder[0] = self.brain.think(user_text, image_path)
                think_thread = threading.Thread(target=_think, daemon=True)
                think_thread.start()

                sounds.play('thinking')
                filler = self._pick_filler(text_lower)
                if filler:
                    tts.play_filler(filler)

                # Wait for inference to finish (may already be done)
                think_thread.join()
                response = response_holder[0]
                tts.speak(response)

                # === POST-TURN: Mode lifecycle and engagement ===
                self._post_turn()

                if self.debug:
                    self._debug_log(user_text, image_path, response, stt_time)

            except KeyboardInterrupt:
                self._shutdown_gracefully()
                break
            except Exception as e:
                print(f"[LIGHTHOUSE] Error in main loop: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                sounds.play('error')
                tts.speak("Oops, I got a little mixed up. What were you saying?")

    def _handle_commands(self, text_lower):
        """Handle meta-commands. Returns True if handled (skip normal processing)."""
        # Exit
        if any(cmd in text_lower for cmd in ['goodbye', 'bye bye', 'see you later', 'quit', 'exit']):
            self._shutdown_gracefully()
            self.running = False
            return True

        # Reset
        if any(cmd in text_lower for cmd in ['start over', 'new topic', 'something else', 'reset']):
            self.brain.reset_session()
            tts.speak("Sure! Want to go on a quest, play a game, or tell me about something cool?")
            return True

        # Help / what can I do
        if any(cmd in text_lower for cmd in ['what can you do', 'help', 'what do we do']):
            tts.speak(
                "We can do lots of things! "
                "I can send you on a quest to find things around your house, "
                "we can make up a silly story together, "
                "you can show me something cool with the camera, "
                "or I can give you a challenge! What sounds fun?"
            )
            return True

        return False

    def _switch_mode(self, mode_key):
        """Switch to a new learning mode with its greeting."""
        greeting = self.brain.enter_mode(mode_key)
        self.turns_since_mode_end = 0
        if greeting:
            sounds.play('startup')
            tts.speak(greeting)

    def _post_turn(self):
        """Handle post-turn logic: mode completion, engagement recovery."""

        # Check if the current mode is done
        if self.brain.mode_is_complete:
            mode_name = MODES[self.brain.active_mode]["name"]
            self.brain.exit_mode()
            self.turns_since_mode_end = 0

            # Offer next activity
            time.sleep(0.3)
            tts.speak("That was awesome! Want to do another quest, a challenge, a story, or something else?")
            return

        # Check for disengagement
        if self.brain.engagement.is_disengaged:
            print("[LIGHTHOUSE] Disengagement detected — suggesting mode switch")
            self.brain.engagement.reset()

            mode_key, suggestion = get_random_mode_suggestion()

            # Don't suggest the same mode they're already in
            if mode_key == self.brain.active_mode:
                mode_key, suggestion = get_random_mode_suggestion()

            tts.speak(suggestion)

            # Auto-enter the suggested mode (they can always redirect)
            self.brain.enter_mode(mode_key)
            return

        # If in free explore and it's been a while, nudge toward a mode
        if self.brain.active_mode == "explore":
            self.turns_since_mode_end += 1
            if self.turns_since_mode_end >= 4 and random.random() < 0.5:
                nudges = [
                    "Hey, want me to give you a challenge?",
                    "I've got a quest for you if you want one!",
                    "Want to make up a silly story together?",
                ]
                # Append the nudge to the response
                tts.speak(random.choice(nudges))
                self.turns_since_mode_end = 0

    def _wants_camera(self, text_lower):
        """Check if the child's speech triggers a camera capture."""
        return any(trigger in text_lower for trigger in config.CAMERA_TRIGGERS)

    def _pick_filler(self, text_lower):
        """Pick a thinking filler phrase to mask LLM latency."""
        if random.random() < 0.3:
            return None

        if self.brain.active_mode == "quest":
            fillers = [
                "Ooh, let me think of your next mission...",
                "Great job, explorer!",
                "The adventure continues...",
            ]
        elif self.brain.active_mode == "story":
            fillers = [
                "Ooh, and then...",
                "Oh wow, what a twist!",
                "The story gets even better...",
            ]
        elif self.brain.active_mode == "challenge":
            fillers = [
                "Let me come up with a good one...",
                "Ready for this?",
            ]
        elif any(w in text_lower for w in ['look', 'see', 'show']):
            fillers = ["Oh, let me see..."]
        elif any(w in text_lower for w in ['why', 'how come', 'how does']):
            fillers = [
                "That's a really good question!",
                "I love that you're asking why!",
            ]
        else:
            fillers = [
                "Hmm, let me think...",
                "That's interesting...",
            ]

        return random.choice(fillers)

    def _debug_log(self, user_text, image_path, response, stt_time):
        """Print debug information after a turn."""
        mode = MODES[self.brain.active_mode]["name"] if self.brain.active_mode else "none"
        print(f"\n[DEBUG] Turn {self.brain.turn_count} | Mode: {mode} "
              f"({self.brain.mode_turn}/{MODES.get(self.brain.active_mode, {}).get('max_turns', '?')})")
        print(f"  Child: \"{user_text}\"")
        print(f"  Image: {'yes' if image_path else 'no'}")
        print(f"  Response: \"{response[:80]}...\"")
        print(f"  Topic: {self.brain.current_topic}")
        print(f"  Engagement score: {self.brain.engagement.disengagement_score}")
        print(f"  STT: {stt_time:.1f}s\n")

    def _shutdown_gracefully(self):
        """Clean shutdown with a farewell."""
        self.running = False
        print("\n[LIGHTHOUSE] Shutting down...")
        tts.speak("It was so fun learning with you today! See you next time, explorer!")
        sounds.play('shutdown')
        time.sleep(1)
        camera.release()
        self.brain.shutdown()
        print("[LIGHTHOUSE] Goodbye!\n")


def main():
    parser = argparse.ArgumentParser(description="Lighthouse — Screen-free AI learning companion")
    parser.add_argument('--no-camera', action='store_true', help='Run without camera')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    lighthouse = Lighthouse(
        use_camera=not args.no_camera,
        debug=args.debug,
    )

    def signal_handler(sig, frame):
        lighthouse._shutdown_gracefully()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    lighthouse.start()


if __name__ == "__main__":
    main()