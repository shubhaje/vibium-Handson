import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from vibium import browser_sync as browser
except ImportError:
    browser = None

from config import CONFIG, HEADLESS


@pytest.mark.skipif(browser is None, reason="vibium not installed")
class TestVibium:
    """Test suite for Vibium browser automation."""
    
    @pytest.fixture
    def vibe(self):
        """Launch browser and cleanup after test."""
        vibe = browser.launch(headless=HEADLESS)
        yield vibe
        vibe.quit()
    
    def test_load_website(self, vibe):
        """Test loading a website."""
        vibe.go("https://example.com")
        assert vibe.url == "https://example.com/"
        print("Loaded example.com")
    
    def test_screenshot(self, vibe):
        """Test taking a screenshot."""
        vibe.go("https://example.com")
        png = vibe.screenshot()
        assert png is not None
        os.makedirs(CONFIG['screenshot']['folder'], exist_ok=True)
        with open(f"{CONFIG['screenshot']['folder']}/screenshot.png", "wb") as f:
            f.write(png)
        print("Saved screenshot.png")
    
    def test_find_element(self, vibe):
        """Test finding an element."""
        vibe.go("https://example.com")
        link = vibe.find("a")
        assert link is not None
        print("Found link:", link.text())
    
    def test_find_multiple_elements(self, vibe):
        """Test finding multiple elements."""
        vibe.go("https://example.com")
        items = vibe.find_all("a")
        assert len(items) > 0
        print(f"Found {len(items)} links on the page.")


def test_config_loaded():
    """Test that configuration loads successfully."""
    assert CONFIG is not None
    assert CONFIG["project_name"] == "vibium"
    assert HEADLESS is True
    print("Configuration loaded successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])