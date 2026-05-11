#!/usr/bin/env python3
"""
Unit tests for Lighthouse optimization fixes — hardware-free.

Covers 5 fixes:
  Fix 1  stt.py    — numpy RMS replaces struct+pure-Python hot-path loop
  Fix 2  camera.py — remove redundant BGR↔RGB↔BGR roundtrip before resize
  Fix 3  brain.py  — sentence trimmer: regex boundaries + fire at > 3 not > 4
  Fix 4  brain.py  — _system_prefix initialized in __init__ (AttributeError guard)
  Fix 5  modes.py  — remove dead response_lengths list from EngagementTracker

Usage:
    python test_unit.py
    python test_unit.py -v
"""
import sys
import math
import struct
import re
import types
import importlib.util
import unittest
import unittest.mock
import numpy as np

# ---------------------------------------------------------------------------
# Stub litert_lm so brain.py can be imported without the real package.
# setdefault leaves a real install untouched.
# ---------------------------------------------------------------------------
_mock_litert = types.ModuleType('litert_lm')
_mock_litert.Engine = unittest.mock.MagicMock()
sys.modules.setdefault('litert_lm', _mock_litert)

import config
from modes import EngagementTracker
from brain import LighthouseBrain


# ===========================================================================
# Fix 1 — stt.py: numpy RMS in the audio hot path
# ===========================================================================

class TestRmsCalculation(unittest.TestCase):
    """numpy RMS must agree with the legacy method and correctly bracket silence vs speech."""

    CHUNK = config.CHUNK_SIZE  # 1024 samples

    def _sine_chunk(self, amplitude):
        """Return CHUNK int16 mono PCM bytes — sine wave at given amplitude."""
        values = [int(amplitude * math.sin(2 * math.pi * i / 64)) for i in range(self.CHUNK)]
        return struct.pack(f'{self.CHUNK}h', *values)

    # ---- reference implementations ----

    def _rms_legacy(self, mono_data):
        """Original implementation (struct.unpack + pure-Python sum)."""
        samples = struct.unpack(f'{self.CHUNK}h', mono_data)
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    def _rms_numpy(self, mono_data):
        """Replacement implementation."""
        arr = np.frombuffer(mono_data, dtype=np.int16)
        return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))

    # ---- tests ----

    def test_silence_is_zero(self):
        data = bytes(self.CHUNK * 2)
        self.assertAlmostEqual(self._rms_numpy(data), 0.0, places=5)

    def test_numpy_matches_legacy_at_multiple_amplitudes(self):
        for amp in (500, 5_000, 20_000):
            with self.subTest(amplitude=amp):
                data = self._sine_chunk(amp)
                self.assertAlmostEqual(
                    self._rms_numpy(data),
                    self._rms_legacy(data),
                    delta=2.0,
                )

    def test_loud_signal_above_silence_threshold(self):
        data = self._sine_chunk(amplitude=20_000)
        self.assertGreater(self._rms_numpy(data), config.SILENCE_THRESHOLD)

    def test_quiet_signal_below_silence_threshold(self):
        data = self._sine_chunk(amplitude=50)
        self.assertLess(self._rms_numpy(data), config.SILENCE_THRESHOLD)

    def test_return_type_is_float(self):
        self.assertIsInstance(self._rms_numpy(self._sine_chunk(1000)), float)

    def test_stereo_downmix_then_rms_matches_mono(self):
        """Stereo left-channel extraction (stt.py line 178) must yield same RMS as raw mono."""
        mono_vals = [int(8000 * math.sin(2 * math.pi * i / 64)) for i in range(self.CHUNK)]
        # Interleave L/R where R is different
        stereo_vals = []
        for v in mono_vals:
            stereo_vals.append(v)       # L
            stereo_vals.append(v // 2)  # R (different)
        stereo_data = struct.pack(f'{self.CHUNK * 2}h', *stereo_vals)

        # Downmix: take every 2nd int16 (left channel)
        extracted = np.frombuffer(stereo_data, dtype=np.int16)[0::2].tobytes()
        mono_data = struct.pack(f'{self.CHUNK}h', *mono_vals)

        self.assertAlmostEqual(self._rms_numpy(extracted), self._rms_numpy(mono_data), delta=2.0)


# ===========================================================================
# Fix 2 — camera.py: BGR↔RGB roundtrip removal
# ===========================================================================

_cv2_available = importlib.util.find_spec('cv2') is not None

@unittest.skipUnless(_cv2_available, "cv2 not installed — skipping camera tests")
class TestCameraColorRoundtrip(unittest.TestCase):
    """Direct resize must produce pixel-identical output to BGR→RGB→resize→RGB→BGR."""

    def _make_frame(self):
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_roundtrip_identical_to_direct_resize(self):
        import cv2
        frame = self._make_frame()
        # Old 3-step path
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, (448, 448), interpolation=cv2.INTER_AREA)
        old_result = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        # New 1-step path
        new_result = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)

        np.testing.assert_array_equal(old_result, new_result)

    def test_output_is_448x448(self):
        import cv2
        frame = self._make_frame()
        result = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
        self.assertEqual(result.shape[:2], (448, 448))

    def test_channel_count_preserved(self):
        import cv2
        frame = self._make_frame()
        result = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
        self.assertEqual(result.shape[2], 3)

    def test_direct_resize_dtype_unchanged(self):
        import cv2
        frame = self._make_frame()
        result = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
        self.assertEqual(result.dtype, np.uint8)


# ===========================================================================
# Fix 3 — brain.py: sentence trimmer
# ===========================================================================

class TestSentenceTrimmer(unittest.TestCase):
    """Fixed trimmer fires at > 3 sentences and recognises .!? boundaries."""

    def _trim(self, text):
        """Replica of the fixed brain.think() trimmer."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) > 3:
            text = ' '.join(sentences[:3])
            if text and text[-1] not in '.!?':
                text += '.'
        return text

    def test_one_sentence_unchanged(self):
        self.assertEqual(self._trim("Hello there."), "Hello there.")

    def test_two_sentences_unchanged(self):
        self.assertEqual(self._trim("Hello there. How are you?"), "Hello there. How are you?")

    def test_three_sentences_unchanged(self):
        text = "One. Two. Three."
        self.assertEqual(self._trim(text), text)

    def test_boundary_exactly_three_not_trimmed(self):
        text = "Hello. How are you? I am fine."
        self.assertEqual(self._trim(text), text)

    def test_four_sentences_trimmed_to_three(self):
        self.assertEqual(self._trim("One. Two. Three. Four."), "One. Two. Three.")

    def test_five_sentences_trimmed_to_three(self):
        self.assertEqual(self._trim("One. Two. Three. Four. Five."), "One. Two. Three.")

    def test_exclamation_marks_count_as_boundaries(self):
        self.assertEqual(self._trim("Wow! Great! Nice! Too much!"), "Wow! Great! Nice!")

    def test_question_marks_count_as_boundaries(self):
        self.assertEqual(self._trim("What? Why? How? When?"), "What? Why? How?")

    def test_mixed_punctuation(self):
        result = self._trim("Hello! How are you? I am fine. Extra sentence.")
        self.assertEqual(result, "Hello! How are you? I am fine.")

    def test_old_bug_regression_four_sentences_now_trimmed(self):
        """Old threshold was > 4 so four-sentence responses slipped through."""
        result = self._trim("One. Two. Three. Four.")
        count = len(re.split(r'(?<=[.!?])\s+', result.strip()))
        self.assertLessEqual(count, 3)

    def test_result_ends_with_terminal_punctuation(self):
        result = self._trim("First. Second. Third. Fourth.")
        self.assertRegex(result, r'[.!?]$')

    def test_empty_string_unchanged(self):
        self.assertEqual(self._trim(""), "")


# ===========================================================================
# Fix 4 — brain.py: _system_prefix initialized in __init__
# ===========================================================================

class TestBrainSystemPrefixInit(unittest.TestCase):
    """_system_prefix must exist after __init__ so _build_prompt is safe before start()."""

    def test_attribute_exists_before_start(self):
        brain = LighthouseBrain()
        self.assertTrue(
            hasattr(brain, '_system_prefix'),
            "_system_prefix must be set in __init__, not only in _send_system_prompt()",
        )

    def test_initial_value_is_none(self):
        brain = LighthouseBrain()
        self.assertIsNone(brain._system_prefix)

    def test_build_prompt_safe_before_start(self):
        brain = LighthouseBrain()
        try:
            result = brain._build_prompt("hello there")
            self.assertIsInstance(result, str)
            self.assertIn("hello there", result)
        except AttributeError as exc:
            self.fail(f"_build_prompt raised AttributeError before start(): {exc}")

    def test_build_prompt_with_active_mode_safe(self):
        brain = LighthouseBrain()
        brain.active_mode = "quest"
        brain.mode_turn = 1
        try:
            brain._build_prompt("I found it!")
        except AttributeError as exc:
            self.fail(f"_build_prompt raised AttributeError with active mode: {exc}")


# ===========================================================================
# Fix 5 — modes.py: remove dead response_lengths from EngagementTracker
# ===========================================================================

class TestEngagementTracker(unittest.TestCase):
    """EngagementTracker correctness after removing the dead response_lengths list."""

    def setUp(self):
        self.tracker = EngagementTracker()

    def test_no_response_lengths_attribute(self):
        self.assertFalse(
            hasattr(self.tracker, 'response_lengths'),
            "response_lengths is dead code — should be removed from EngagementTracker",
        )

    def test_initially_not_disengaged(self):
        self.assertFalse(self.tracker.is_disengaged)

    def test_initial_score_is_zero(self):
        self.assertEqual(self.tracker.disengagement_score, 0)

    def test_disengaged_phrase_adds_two(self):
        self.tracker.record_turn("i don't know")
        self.assertEqual(self.tracker.disengagement_score, 2)

    def test_i_dunno_adds_two(self):
        self.tracker.record_turn("i dunno")
        self.assertEqual(self.tracker.disengagement_score, 2)

    def test_short_nonmatching_response_adds_one(self):
        # "up there" — 2 words, not a substring of any DISENGAGED_PHRASES entry
        self.tracker.record_turn("up there")
        self.assertEqual(self.tracker.disengagement_score, 1)

    def test_engaged_response_decays_score(self):
        self.tracker.disengagement_score = 2
        # Sentence chosen to avoid pre-existing substring-match false positives
        # (e.g. "no" inside "dinosaurs", "know", "now" all match DISENGAGED_PHRASES)
        self.tracker.record_turn("I totally love learning about outer space and the galaxy")
        self.assertEqual(self.tracker.disengagement_score, 1)

    def test_score_floored_at_zero(self):
        self.tracker.disengagement_score = 0
        self.tracker.record_turn("I absolutely love learning about the stars today")
        self.assertEqual(self.tracker.disengagement_score, 0)

    def test_is_disengaged_at_score_three(self):
        self.tracker.disengagement_score = 3
        self.assertTrue(self.tracker.is_disengaged)

    def test_not_disengaged_below_threshold(self):
        self.tracker.disengagement_score = 2
        self.assertFalse(self.tracker.is_disengaged)

    def test_reset_clears_score(self):
        self.tracker.disengagement_score = 5
        self.tracker.reset()
        self.assertEqual(self.tracker.disengagement_score, 0)

    def test_reset_leaves_tracker_not_disengaged(self):
        self.tracker.disengagement_score = 5
        self.tracker.reset()
        self.assertFalse(self.tracker.is_disengaged)

    def test_reset_does_not_reference_removed_attribute(self):
        """reset() must not try to call .clear() on the removed response_lengths list."""
        try:
            self.tracker.reset()
        except AttributeError as exc:
            self.fail(f"reset() raised AttributeError (likely response_lengths.clear()): {exc}")

    def test_multiple_turns_accumulate_correctly(self):
        self.tracker.record_turn("up there")       # +1 → 1
        self.tracker.record_turn("i don't know")   # +2 → 3
        self.assertTrue(self.tracker.is_disengaged)

    def test_recovery_from_disengaged(self):
        self.tracker.disengagement_score = 3
        # Sentences chosen to avoid pre-existing substring-match false positives
        self.tracker.record_turn("Tell me all about the solar system and how Jupiter formed")
        self.tracker.record_turn("That is so cool I want to discover more about Saturn")
        self.tracker.record_turn("Let me find objects that are blue and red all around")
        self.assertFalse(self.tracker.is_disengaged)


# ===========================================================================
# Fix 6 — modes.py: DISENGAGED_PHRASES substring false positives
# ===========================================================================

class TestEngagementTrackerWordBoundary(unittest.TestCase):
    """DISENGAGED_PHRASES matching must use word boundaries, not bare substring search.

    Known false positives with the bare-substring approach:
      "no"  fires on  "dinosaurs", "know", "now", "knob", "ignore" …
    After fix: single-word phrases use \\b word-boundary matching.
    """

    def setUp(self):
        self.tracker = EngagementTracker()

    def test_dinosaurs_does_not_trigger_no(self):
        """'dinosaurs' contains the substring 'no' — must not count as disengaged."""
        self.tracker.record_turn("I love learning about dinosaurs")
        # With the bare-substring bug: disengagement_score == 2 (false positive)
        # After fix: disengagement_score == 0 (engaged, long response)
        self.assertEqual(self.tracker.disengagement_score, 0,
                         "Substring 'no' inside 'dinosaurs' must not trigger disengagement")

    def test_know_does_not_trigger_no(self):
        self.tracker.record_turn("I want to know more about planets")
        self.assertEqual(self.tracker.disengagement_score, 0,
                         "Substring 'no' inside 'know' must not trigger disengagement")

    def test_right_now_does_not_trigger_no(self):
        self.tracker.record_turn("I want to do a quest right now")
        self.assertEqual(self.tracker.disengagement_score, 0,
                         "Substring 'no' inside 'now' must not trigger disengagement")

    def test_standalone_no_still_triggers(self):
        """A bare 'no' in isolation must still fire."""
        self.tracker.record_turn("no")
        self.assertEqual(self.tracker.disengagement_score, 2)

    def test_sure_does_not_trigger_on_reassure(self):
        """'sure' must not match inside 'reassure'."""
        self.tracker.record_turn("That will reassure me and make me feel better")
        self.assertEqual(self.tracker.disengagement_score, 0,
                         "Substring 'sure' inside 'reassure' must not trigger disengagement")

    def test_standalone_sure_still_triggers(self):
        self.tracker.record_turn("sure")
        self.assertEqual(self.tracker.disengagement_score, 2)

    def test_stop_does_not_trigger_on_stopping(self):
        self.tracker.record_turn("I am stopping to look at all these objects")
        self.assertEqual(self.tracker.disengagement_score, 0,
                         "Substring 'stop' inside 'stopping' must not trigger disengagement")

    def test_standalone_stop_still_triggers(self):
        self.tracker.record_turn("stop")
        self.assertEqual(self.tracker.disengagement_score, 2)


# ===========================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
