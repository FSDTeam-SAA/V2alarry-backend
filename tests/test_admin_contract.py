import unittest
from datetime import datetime, timezone

from app.api.v1.admin.dashboard import utc_week_bounds
from app.schemas.admin import AdminCoachingSummaryResponse


class AdminContractTests(unittest.TestCase):
    def test_utc_week_bounds_start_on_monday_and_cover_two_calendar_weeks(self):
        current_start, current_end, previous_start = utc_week_bounds(
            datetime(2026, 9, 17, 14, 30, tzinfo=timezone.utc)
        )

        self.assertEqual(current_start.isoformat(), "2026-09-14T00:00:00+00:00")
        self.assertEqual(current_end.isoformat(), "2026-09-21T00:00:00+00:00")
        self.assertEqual(previous_start.isoformat(), "2026-09-07T00:00:00+00:00")

    def test_admin_summary_contract_has_no_transcript_or_working_state_fields(self):
        fields = AdminCoachingSummaryResponse.model_fields
        self.assertNotIn("messages", fields)
        self.assertNotIn("transcript", fields)
        self.assertNotIn("working_state", fields)
        self.assertIn("generation_status", fields)
        self.assertIn("commitment", fields)


if __name__ == "__main__":
    unittest.main()
