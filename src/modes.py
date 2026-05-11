"""Learning modes — structured interaction patterns that keep kids engaged.

The key insight: young children lose interest in free-form Q&A fast.
They stay engaged with MISSIONS, CHALLENGES, and SURPRISES that give
them a reason to move, touch, and explore the physical world.

Each mode has:
- An activation prompt (how the mode starts)
- A turn structure (what happens each turn)
- Engagement hooks (what keeps them going)
- A natural exit (when the mode ends)
"""
import random
import re


# ====================================================================
# MODE DEFINITIONS
# ====================================================================

# Each mode is a dict with prompts the brain uses to guide Gemma.
# The brain injects the active mode's prompt into the conversation.

MODES = {

    "quest": {
        "name": "Quest",
        "description": "A 3-step scavenger hunt around the house",
        "icon": "compass",  # for future UI
        "greeting": (
            "Quest time! I've got a mission for you. "
            "Are you ready, explorer?"
        ),
        "system_context": """You are running a QUEST — a 3-step scavenger hunt.

RULES FOR QUESTS:
- You give the child exactly 3 challenges, one at a time
- Each challenge asks them to FIND, TOUCH, or BRING BACK a real object
- After each challenge, celebrate what they found and give the next one
- Challenges should be related to a single theme (animals, colors, shapes, textures, etc.)
- After all 3 challenges, give them a fun "explorer title" based on what they found

PACING: Give ONE challenge at a time. Wait for the child to respond before the next.
TONE: Excited, adventurous, like a treasure hunt narrator.
LENGTH: 1-2 sentences per response. Short and punchy.

Quest structure:
- Turn 1: Pick a theme and give Challenge 1 (e.g., "Find something SOFT in your room!")
- Turn 2: Celebrate + Challenge 2 (e.g., "Now find something that makes a sound!")
- Turn 3: Celebrate + Challenge 3 (e.g., "Last one — find the SMALLEST thing you can!")
- Turn 4: Celebrate all 3, give a fun title (e.g., "You are now the Texture Explorer!")
""",
        "max_turns": 5,
    },

    "show_and_tell": {
        "name": "Show & Tell",
        "description": "Child shows objects, Lighthouse reacts and asks questions",
        "icon": "camera",
        "greeting": (
            "Ooh, I love Show and Tell! "
            "Show me something interesting and I'll tell you what I notice!"
        ),
        "system_context": """You are running SHOW AND TELL — the child shows objects to the camera.

RULES FOR SHOW AND TELL:
- When you see an image, describe 2-3 SPECIFIC things you notice (colors, shapes, details)
- Ask ONE curious question about what you see
- Connect what you see to something the child might know
- If they show a drawing, comment on specific choices they made (colors, shapes) — NEVER criticize
- Encourage them to show you more: "What else do you have?"

TONE: Genuinely curious and impressed. React like you've never seen anything so interesting.
LENGTH: 2 sentences describing what you see + 1 question. That's it.
""",
        "max_turns": 8,
    },

    "story": {
        "name": "Story Time",
        "description": "Collaborative storytelling where child and Lighthouse take turns",
        "icon": "book",
        "greeting": (
            "Let's make up a story together! "
            "I'll start, and then you add what happens next. Ready?"
        ),
        "system_context": """You are running STORY TIME — a collaborative story with the child.

RULES FOR STORY TIME:
- You and the child take turns adding to a story
- YOUR turns: add 1-2 sentences that set up an exciting moment or choice
- Always end YOUR turn with a question: "What happens next?" or "What does [character] do?"
- Build on EVERYTHING the child says — never ignore or redirect their ideas
- Include silly, surprising elements (talking animals, magic, funny sounds)
- Use sound effects in your narration ("WHOOOOSH! The dragon flew over the castle!")
- After 4-5 exchanges, start wrapping up: "And then, for the big ending..."

TONE: Dramatic, playful, full of wonder. Use different "voices" for characters.
LENGTH: 1-2 sentences of story + 1 question. Keep it moving fast.
""",
        "max_turns": 8,
    },

    "challenge": {
        "name": "Quick Challenge",
        "description": "A fast, single hands-on challenge with a countdown",
        "icon": "lightning",
        "greeting": (
            "Challenge time! I've got a quick one for you. "
            "Think you can do it?"
        ),
        "system_context": """You are running a QUICK CHALLENGE — a single, fun, timed activity.

RULES FOR CHALLENGES:
- Give ONE specific, physical challenge the child can do RIGHT NOW
- The challenge must use objects found in a typical home
- Include a playful "countdown" or "timer" element ("Can you do it before I count to ten?")
- Challenges should be ACTIVE: stack, sort, build, draw, find, arrange, count
- After they complete it, celebrate enthusiastically and offer ONE more: "Want another challenge?"

GOOD CHALLENGES:
- "Stack 5 things on top of each other — tallest tower wins!"
- "Find 4 things that are the same color. Go!"
- "Draw the silliest animal you can think of, you have one minute!"
- "Arrange your stuffed animals from smallest to biggest!"

BAD CHALLENGES (never suggest):
- Anything involving water, scissors, heights, going outside, or electronics

TONE: High energy, game-show host vibes. Make it feel exciting and urgent.
LENGTH: 1-2 sentences. Fast and punchy.
""",
        "max_turns": 4,
    },

    "explore": {
        "name": "Free Explore",
        "description": "Open-ended conversation about anything the child is curious about",
        "icon": "magnifying-glass",
        "greeting": None,  # No special greeting, just flows naturally
        "system_context": """You are in FREE EXPLORE mode — the child leads the conversation.

RULES FOR FREE EXPLORE:
- Follow the child's curiosity wherever it goes
- Answer their questions simply, then ask a follow-up
- If they seem stuck, offer 2 fun choices: "Want to learn about space, or about bugs?"
- Every 3rd response, suggest something PHYSICAL to do related to the topic
- If the conversation stalls (short responses, "I don't know"), switch to a Quest or Challenge

TONE: Curious, warm, responsive.
LENGTH: 1-2 sentences + 1 question or activity suggestion.
""",
        "max_turns": 99,  # No hard limit for explore
    },
}


# ====================================================================
# MODE DETECTION
# ====================================================================

# Phrases that trigger specific modes (matched case-insensitive, substring)
MODE_TRIGGERS = {
    "quest": [
        "quest", "mission", "adventure", "treasure hunt",
        "scavenger", "let's find", "go find",
    ],
    "show_and_tell": [
        "show and tell", "show you", "look at this", "look what i",
        "see this", "see what i", "check this out",
    ],
    "story": [
        "story", "tell me a story", "make up a story",
        "once upon a time", "let's pretend",
    ],
    "challenge": [
        "challenge", "dare", "game", "play a game",
        "something to do", "i'm bored", "im bored",
        "what should i do", "what can i do",
    ],
}


def detect_mode(text_lower):
    """Detect which mode the child's speech is requesting.

    Returns mode key (str) or None for free explore.
    """
    for mode_key, triggers in MODE_TRIGGERS.items():
        if any(trigger in text_lower for trigger in triggers):
            return mode_key
    return None


def get_random_mode_suggestion():
    """Get a suggestion to offer when the child seems disengaged.

    Returns (mode_key, spoken_suggestion).
    """
    suggestions = [
        ("quest", "Hey, want to go on a quest? I'll give you a mission to explore your house!"),
        ("challenge", "I've got a challenge for you! Want to try it?"),
        ("story", "Want to make up a silly story together? I'll start!"),
        ("show_and_tell", "Show me something cool! Grab your favorite thing and let me see it."),
    ]
    return random.choice(suggestions)


# ====================================================================
# ENGAGEMENT TRACKING
# ====================================================================

class EngagementTracker:
    """Tracks engagement signals to detect when a child is losing interest.

    Signals of disengagement:
    - Very short responses (1-2 words)
    - Long pauses (detected by caller)
    - Repeated "I don't know" / "nothing" / "no"
    - Decreasing response length over time
    """

    DISENGAGED_PHRASES = [
        "i don't know", "i dunno", "nothing", "no", "nah",
        "whatever", "okay", "sure", "fine", "boring",
        "i'm bored", "im bored", "stop", "be quiet",
    ]

    def __init__(self):
        self.disengagement_score = 0  # 0 = fully engaged, 3+ = suggest mode switch

    def record_turn(self, child_text):
        """Record the child's response and update engagement signals."""
        words = child_text.strip().split()
        text_lower = child_text.lower().strip()

        def _matches(phrase):
            if ' ' in phrase:
                return phrase in text_lower
            return bool(re.search(r'\b' + re.escape(phrase) + r'\b', text_lower))

        if any(_matches(phrase) for phrase in self.DISENGAGED_PHRASES):
            self.disengagement_score += 2
        elif len(words) <= 2:
            self.disengagement_score += 1
        else:
            # Engaged response — decay the score
            self.disengagement_score = max(0, self.disengagement_score - 1)

    @property
    def is_disengaged(self):
        """Returns True if the child seems to be losing interest."""
        return self.disengagement_score >= 3

    def reset(self):
        """Reset tracking for a new mode."""
        self.disengagement_score = 0
