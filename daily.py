import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from plot import plot_constellations

"""
Daily automated sky generation script
Generates night sky plots for 8 cities across all continents
Each city shown at midnight local time
"""

print("=" * 60)
print("DAILY SKY GENERATION")
print("=" * 60)

# Get current date (will use midnight for each timezone)
today = datetime.now()

# City configurations: (name, timezone)
cities = [
    ("Los Angeles", "America/Los_Angeles"),
    ("New York", "America/New_York"),
    ("Ushuaia", "America/Argentina/Ushuaia"),
    ("Berlin", "Europe/Berlin"),
    ("Cape Town", "Africa/Johannesburg"),
    ("Beijing", "Asia/Shanghai"),
    ("Singapore", "Asia/Singapore"),
    ("Sydney", "Australia/Sydney"),
]

for city, timezone in cities:
    print(f"\n{'='*60}")
    print(f"Generating sky for {city} at midnight local time")
    print(f"Timezone: {timezone}")

    # Set time to next midnight (tomorrow night) local time for this city
    now_local = datetime.now(ZoneInfo(timezone))
    tomorrow = now_local + timedelta(days=1)
    local_midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Next night's sky: {local_midnight.strftime('%Y-%m-%d %H:%M %Z')}")

    # Generate the plot
    plot_constellations(
        place=city, time=local_midnight, mode="visible", limiting_magnitude=6.0
    )

    print(f"✓ {city} complete")
    time.sleep(1)

print("\n" + "=" * 60)
print("All daily sky generations complete!")
print(f"Generated {len(cities)} city sky plots")
print("Check the ./images directory")
print("=" * 60)
