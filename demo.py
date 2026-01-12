from datetime import datetime, timezone

from plot import plot_constellations

"""
Showcase file for plot_constellations function
Demonstrates different parameters and use cases
"""

print("=" * 60)
print("PLOT_CONSTELLATIONS SHOWCASE")
print("=" * 60)
plot_constellations()

# Example 1: Basic - current visible sky
print("\n1. Basic example - Reykjavik visible sky (current time)")
plot_constellations(place="Reykjavik")

# Example 2: Show both visible and non-visible sides
print("\n2. Both sides view - Reykjavik")
plot_constellations(place="Reykjavik", mode="both")

# Example 3: Non-visible constellations only
print("\n3. Non-visible constellations - Sydney")
plot_constellations(place="Sydney", mode="nonvisible")

# Example 4: Summer solstice in Los Angeles
print("\n4. Summer solstice - Los Angeles (June 21, 2025 midnight UTC)")
summer_time = datetime(2025, 6, 21, 0, 0, tzinfo=timezone.utc)
plot_constellations(place="Los Angeles", time=summer_time)

# Example 5: Winter solstice in Los Angeles
print("\n5. Winter solstice - Los Angeles (December 21, 2025 midnight UTC)")
winter_time = datetime(2025, 12, 21, 0, 0, tzinfo=timezone.utc)
plot_constellations(place="Los Angeles", time=winter_time)

# Example 6: Light-polluted city (fewer stars visible)
print("\n6. Light pollution example - Tokyo (magnitude 4.0)")
plot_constellations(place="Tokyo", limiting_magnitude=4.0)

# Example 7: Exceptionally dark sky (more stars visible)
print("\n7. Dark sky example - Reykjavik (magnitude 7.0)")
plot_constellations(place="Reykjavik", limiting_magnitude=7.0)

# Example 8: Custom filename
print("\n8. Custom filename example - Cape Town")
plot_constellations(place="Cape Town", fname="my_custom_sky.png")

# Example 9: Combined parameters - winter in Berlin, both sides, dark sky
print("\n9. Combined parameters - Berlin winter, both sides, dark sky")
plot_constellations(
    place="Berlin",
    mode="both",
    limiting_magnitude=6.5,
    time=winter_time,
    fname="berlin_winter_both.png",
)

print("\n" + "=" * 60)
print("All examples complete! Check the ./images directory")
print("=" * 60)
