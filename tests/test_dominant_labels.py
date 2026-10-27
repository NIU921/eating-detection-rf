import unittest

from dominant_hand.config import Config
from dominant_hand.labels import window_label


class ConfirmedBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.meals = [dict(start=120000, end=300000, dominant=True),
                      dict(start=600000, end=720000, dominant=False)]

    def test_confirmed_non_eating_record_is_negative(self):
        self.assertEqual(window_label(0, 10000, [], self.cfg)[0], 0)

    def test_background_on_both_sides_of_meal_is_negative(self):
        self.assertEqual(window_label(0, 10000, self.meals, self.cfg)[0], 0)
        self.assertEqual(window_label(400000, 410000, self.meals, self.cfg)[0], 0)

    def test_meal_interior_is_positive(self):
        self.assertEqual(window_label(150000, 160000, self.meals, self.cfg)[0], 1)

    def test_training_buffers_are_ignored(self):
        for a,b in [(110000,120000),(300000,310000)]:
            self.assertEqual(window_label(a,b,self.meals,self.cfg),(-1,"boundary_buffer"))

    def test_non_dominant_meal_is_never_negative(self):
        self.assertEqual(window_label(630000,640000,self.meals,self.cfg)[0],-1)

    def test_partly_overlapping_meal_is_not_negative(self):
        self.assertEqual(window_label(115000,125000,self.meals,self.cfg)[0],-1)

    def test_conflict_overrides_positive(self):
        meals=self.meals+[dict(start=150000,end=160000,dominant=False)]
        self.assertEqual(window_label(150000,160000,meals,self.cfg)[0],-1)

    def test_optional_explicit_restriction_can_disable_background(self):
        self.assertEqual(window_label(0,10000,[],self.cfg,[])[0],-1)
        self.assertEqual(window_label(0,10000,[],self.cfg,[(0,10000)])[0],0)


if __name__ == "__main__":
    unittest.main()
