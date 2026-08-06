import json
import unittest

from mlagents_envs.side_channel import IncomingMessage, OutgoingMessage

from training_utils import (
    CustomChannel,
    attach_collision_events_to_running_transitions,
)


def deliver(channel, payload):
    outgoing = OutgoingMessage()
    outgoing.write_string(payload)
    channel.on_message_received(IncomingMessage(outgoing.buffer))


class CollisionSideChannelTests(unittest.TestCase):
    def test_collision_events_are_attached_to_the_matching_episode(self):
        channel = CustomChannel()
        collision_event = {
            'id': 2,
            'seed': 101,
            'physics_step': 17,
            'collision_index': 1,
            'collider_type': 'wall',
            'position_x': 1.25,
            'position_z': -0.5,
            'relative_speed': 0.8,
        }

        deliver(
            channel,
            channel.COLLISION_EVENT_TOKEN + '|' + json.dumps(collision_event),
        )
        deliver(
            channel,
            channel.END_EPISODE_TOKEN
            + '|'
            + json.dumps({'id': 2, 'seed': 202, 'collisions': 0}),
        )
        deliver(
            channel,
            channel.END_EPISODE_TOKEN
            + '|'
            + json.dumps({'id': 2, 'seed': 101, 'collisions': 1}),
        )

        self.assertEqual(channel.stop_msg_queue[0]['collision_events'], [])
        self.assertEqual(
            channel.stop_msg_queue[1]['collision_events'],
            [collision_event],
        )
        self.assertEqual(channel._collision_events, {})

    def test_clear_queue_removes_pending_collision_events(self):
        channel = CustomChannel(capture_collision_steps=True)
        deliver(
            channel,
            channel.COLLISION_EVENT_TOKEN
            + '|'
            + json.dumps({'id': 0, 'seed': 1, 'physics_step': 5}),
        )

        channel.clear_queue()

        self.assertEqual(channel.start_msg_queue, [])
        self.assertEqual(channel.stop_msg_queue, [])
        self.assertEqual(channel._collision_events, {})

    def test_event_is_attached_to_transition_and_inner_step(self):
        event = {'id': 4, 'seed': 12, 'physics_step': 9}
        observations = {31: [None, None, None, 0, 4]}
        running_episodes = {
            31: [{
                'collision': False,
                'collision_count': 0,
                'collision_events': [],
                'inner_steps': [{
                    'collision': False,
                    'collision_events': [],
                }],
            }]
        }

        pending = attach_collision_events_to_running_transitions(
            [event],
            observations,
            running_episodes,
        )

        transition = running_episodes[31][0]
        self.assertEqual(pending, [])
        self.assertTrue(transition['collision'])
        self.assertEqual(transition['collision_count'], 1)
        self.assertEqual(transition['collision_events'], [event])
        self.assertTrue(transition['inner_steps'][0]['collision'])
        self.assertEqual(
            transition['inner_steps'][0]['collision_events'],
            [event],
        )


if __name__ == '__main__':
    unittest.main()
