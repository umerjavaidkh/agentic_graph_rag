"""Unit tests for per-user thread_id scoping."""
from __future__ import annotations

import unittest

from src.auth.thread_scope import scoped_thread_id


class TestThreadScope(unittest.TestCase):
    def test_default_suffix(self):
        self.assertEqual(scoped_thread_id("101180639787655800606", None), "101180639787655800606:default")

    def test_client_suffix_only(self):
        self.assertEqual(
            scoped_thread_id("user_a", "abc123"),
            "user_a:abc123",
        )

    def test_strips_foreign_user_prefix(self):
        """Client cannot attach to another user's namespace."""
        self.assertEqual(
            scoped_thread_id("user_b", "user_a:secret-thread"),
            "user_b:secret-thread",
        )

    def test_accepts_full_id_but_rebinds_user(self):
        self.assertEqual(
            scoped_thread_id("user_b", "user_b:my-chat"),
            "user_b:my-chat",
        )

    def test_invalid_suffix_falls_back_to_default(self):
        self.assertEqual(scoped_thread_id("u1", "../../etc"), "u1:default")


if __name__ == "__main__":
    unittest.main()
