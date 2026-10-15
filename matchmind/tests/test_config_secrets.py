"""Settings never print API keys (a failing test once dumped every key into the log)."""
from matchmind.config import Settings


def test_repr_masks_api_keys_and_database_password():
    s = Settings(GROQ_API_KEY="dummy-secret-123", DATABASE_URL="postgresql://u:pw-secret@h/db", _env_file=None)
    text = repr(s) + str(s)
    assert "dummy-secret-123" not in text and "pw-secret" not in text
    assert "groq_api_key" in text                          # still shows the field exists
    assert s.groq_api_key == "dummy-secret-123"            # the value itself is untouched
