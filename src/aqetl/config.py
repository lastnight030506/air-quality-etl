"""Static configuration: cities, pollutants, thresholds for API access."""

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    key: str
    name: str
    latitude: float
    longitude: float


# 25 km search radius around each city centre is the OpenAQ maximum for a
# coordinate search that still returns city-scale stations.
# Only these two have live OpenAQ stations. All 63 Vietnamese locations were
# checked: 11 reported inside 90 days and every one sits in Hanoi or Ho Chi Minh
# City. Da Nang returns exactly zero locations within 25 km, so listing it would
# draw an empty series and imply coverage that does not exist.
CITIES: tuple[City, ...] = (
    City("hanoi", "Hanoi", 21.0285, 105.8542),
    City("hcmc", "Ho Chi Minh City", 10.8231, 106.6297),
)

SEARCH_RADIUS_M = 25_000

# OpenAQ parameter ids. Names in the payload are the internal ones, not the
# display names, so we key on the id and map to our own canonical name.
PARAMETERS: dict[int, str] = {
    1: "pm10",
    2: "pm25",
    7: "no2",
    10: "o3",
}

PARAMETER_ORDER: tuple[str, ...] = ("pm25", "pm10", "no2", "o3")

ATTRIBUTION = "Data source: OpenAQ (https://openaq.org), CC BY 4.0 unless stated otherwise."

# Retain a rolling window so the committed store stays small.
RETENTION_DAYS = 400

# OpenAQ free tier: 60 requests/minute, 2000/hour.
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_MAX_PER_PAGE = 1000
OPENAQ_TIMEOUT_S = 30.0
