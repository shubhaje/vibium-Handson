from vibium import browser_sync as browser

# Launch a browser (you'll see it open!)
vibe = browser.launch(headless=True)

# Go to a website
vibe.go("https://example.com")
print("Loaded example.com")

# Take a screenshot
png = vibe.screenshot()
with open("screenshot.png", "wb") as f:
    f.write(png)
print("Saved screenshot.png")

# Find and click the link
link = vibe.find("a")
print("Found link:", link.text())
link.click()
print("Clicked!")
#Find multiple links    
items = vibe.find_all("a")
print(f"Found {len(items)} links on the page.")
# Close the browser
vibe.quit()
print("Done!")