# Configuration for Vibium project

CONFIG = {
    "project_name": "vibium",
    "version": "1.0.0",
    "description": "Browser automation testing with vibium",
    "browser": {
        "headless": True,
        "timeout": 30,
        "width": 1280,
        "height": 720
    },
    "screenshot": {
        "folder": "screenshots",
        "format": "png",
        "quality": 95
    }
}

# Access config values
PROJECT_NAME = CONFIG["project_name"]
VERSION = CONFIG["version"]
HEADLESS = CONFIG["browser"]["headless"]
TIMEOUT = CONFIG["browser"]["timeout"]
SCREENSHOT_FOLDER = CONFIG["screenshot"]["folder"]