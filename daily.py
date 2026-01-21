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

# City configurations: (name, timezone, lat, lon)
cities = [
    ("Los Angeles", "America/Los_Angeles", 34.0522, -118.2437),
    ("New York", "America/New_York", 40.7128, -74.0060),
    ("Ushuaia", "America/Argentina/Ushuaia", -54.8019, -68.3030),
    ("Berlin", "Europe/Berlin", 52.5200, 13.4050),
    ("Cape Town", "Africa/Johannesburg", -33.9249, 18.4241),
    ("Beijing", "Asia/Shanghai", 39.9042, 116.4074),
    ("Singapore", "Asia/Singapore", 1.3521, 103.8198),
    ("Sydney", "Australia/Sydney", -33.8688, 151.2093),
]

for city, timezone, lat, lon in cities:
    print(f"\n{'='*60}")
    print(f"Generating sky for {city} at midnight local time")
    print(f"Timezone: {timezone}")

    # Set time to next midnight (tomorrow night) local time for this city
    now_local = datetime.now(ZoneInfo(timezone))
    tomorrow = now_local + timedelta(days=1)
    local_midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Next night's sky: {local_midnight.strftime('%Y-%m-%d %H:%M %Z')}")

    # Generate the plot (save to daily/ subfolder)
    filename = f"daily/{city.replace(' ', '_').lower()}_visible.png"
    plot_constellations(
        place=city,
        time=local_midnight,
        mode="visible",
        limiting_magnitude=6.0,
        fname=filename,
        lat=lat,
        lon=lon,
    )

    print(f"✓ {city} complete")
    time.sleep(2)

print("\n" + "=" * 60)
print("All daily sky generations complete!")
print(f"Generated {len(cities)} city sky plots")
print("Check the ./images directory")
print("=" * 60)
