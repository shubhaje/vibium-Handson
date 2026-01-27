# Vibium - Browser Automation Testing

Vibium is a Python-based browser automation library for testing and web scraping.

## Project Structure

```
vibium/
├── config.py           # Configuration settings
├── test_vibium.py      # Test suite
├── README.md           # This file
└── screenshots/        # Output directory for screenshots
```

## Configuration

Edit `config.py` to customize:
- Browser headless mode
- Timeout settings
- Screenshot quality
- Output folder

## Running Tests

```bash
# Install dependencies
pip install vibium selenium python-dotenv pytest

# Run all tests
python -m pytest test_vibium.py -v

# Run specific test
python -m pytest test_vibium.py::TestVibium::test_load_website -v
```

## GitHub Actions CI/CD

Automated tests run on:
- Push to main branch
- Pull requests
- Python versions: 3.8, 3.9, 3.10
- Platform: Windows

## Features

- Launch browser (headless or headed)
- Navigate to websites
- Take screenshots
- Find and interact with elements
- Multi-element selection

## License

MIT
