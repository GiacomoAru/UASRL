import unittest

from action_difference import mean_successive_action_diff_for_episode
from uncertainty_utils import saved_transition_values


class RichTransitionSchemaTests(unittest.TestCase):
    def test_rich_transition_can_be_converted_to_model_vector(self):
        transition = {
            'obs': [1.0, 2.0, 3.0],
            'action': [0.4, -0.2],
            'collision': True,
            'collision_events': [{'physics_step': 7}],
        }

        self.assertEqual(
            saved_transition_values(transition),
            [1.0, 2.0, 3.0, 0.4, -0.2],
        )

    def test_action_analysis_accepts_rich_transitions(self):
        episode = [
            {'obs': [0.0], 'action': [0.0, 0.0]},
            {'obs': [1.0], 'action': [3.0, 4.0]},
        ]

        self.assertEqual(mean_successive_action_diff_for_episode(episode), 5.0)


if __name__ == '__main__':
    unittest.main()
