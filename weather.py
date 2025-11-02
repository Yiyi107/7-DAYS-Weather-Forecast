"""Fetch temperatures from Open-Meteo and compute simple NumPy statistics.

Usage: set latitude/longitude and optional start/end dates below or run
as a script to fetch daily mean temperatures and compute max/min/avg.

Dependencies: requests, requests-cache (optional), retry-requests (optional), numpy
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import List, Optional

import numpy as np
import requests
import requests_cache

URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def fetch_daily_mean_temperature(lat: float, lon: float, start: str, end: str, use_cache: bool = True) -> tuple[List[float], List[float], List[float], List[str]]:
	"""Fetch daily mean temperature (2m) from Open-Meteo between start and end (YYYY-MM-DD).

	Returns a list of temperatures in Celsius. Raises RuntimeError on bad response.
	"""
	if use_cache:
		# short-lived cache to avoid repeated API hits during development
		requests_cache.install_cache("openmeteo_cache", expire_after=300)

	params = {
		"latitude": lat,
		"longitude": lon,
		"start_date": start,
		"end_date": end,
		# request daily max/min temperatures
		"daily": "temperature_2m_max,temperature_2m_min",
		"timezone": "UTC",
	}

	resp = requests.get(URL, params=params, timeout=10)
	if resp.status_code != 200:
		raise RuntimeError(f"Open-Meteo request failed: {resp.status_code} {resp.text}")

	data = resp.json()

	daily = data.get("daily", {})
	tmax = daily.get("temperature_2m_max")
	tmin = daily.get("temperature_2m_min")
	dates = daily.get("time")
	if not tmax or not tmin or not dates:
		raise RuntimeError("No daily temperature data returned from Open-Meteo")

	# compute daily mean from (max + min) / 2 for each day
	daily_means: List[float] = [float((mx + mn) / 2.0) for mx, mn in zip(tmax, tmin)]
	# also return daily max/min alongside means by packing into a tuple
	return daily_means, list(tmax), list(tmin), list(dates)


def geocode_city(city: str) -> tuple[float, float]:
	"""Resolve a city name to (lat, lon) using Open-Meteo geocoding API.

	Raises RuntimeError if no results.
	"""
	params = {"name": city, "count": 1}
	resp = requests.get(GEOCODE_URL, params=params, timeout=10)
	if resp.status_code != 200:
		raise RuntimeError(f"Geocoding request failed: {resp.status_code} {resp.text}")
	data = resp.json()
	results = data.get("results")
	if not results:
		raise RuntimeError(f"No geocoding results for city: {city}")
	loc = results[0]
	return float(loc["latitude"]), float(loc["longitude"]) 


def compute_stats(temps_celsius: List[float]) -> dict:
	arr = np.array(temps_celsius, dtype=float)
	return {
		"highest_c": float(np.max(arr)),
		"lowest_c": float(np.min(arr)),
		"average_c": float(np.mean(arr)),
		"fahrenheit": (arr * 9 / 5) + 32,
		"days_above_20c": int((arr > 20).sum()),
		"count": int(arr.size),
	}


def main(argv: Optional[list[str]] = None) -> int:
	p = argparse.ArgumentParser(description="Fetch temperatures from Open-Meteo and compute stats")
	p.add_argument("--lat", type=float, help="Latitude")
	p.add_argument("--lon", type=float, help="Longitude")
	p.add_argument("--city", type=str, help="City name (will be geocoded). If provided, --lat/--lon are ignored.")
	# default to next 7 days (today..today+6)
	p.add_argument("--start", type=str, help="Start date YYYY-MM-DD", default=dt.date.today().isoformat())
	p.add_argument("--end", type=str, help="End date YYYY-MM-DD", default=(dt.date.today() + dt.timedelta(days=6)).isoformat())
	p.add_argument("--no-cache", action="store_true", help="Disable requests-cache")

	args = p.parse_args(argv)

	# resolve location: priority city -> lat/lon -> interactive prompt
	if args.city:
		lat, lon = geocode_city(args.city)
	elif args.lat is not None and args.lon is not None:
		lat, lon = args.lat, args.lon
	else:
		# interactive prompt
		city = input("Enter a city name (e.g. London): ").strip()
		lat, lon = geocode_city(city)

	try:
		temps, tmax, tmin, dates = fetch_daily_mean_temperature(lat, lon, args.start, args.end, use_cache=not args.no_cache)
	except Exception as e:
		print("Error fetching temperatures:", e, file=sys.stderr)
		return 2

	if len(temps) == 0:
		print("No temperature points returned.")
		return 1

	stats = compute_stats(temps)

	print(f"Dates: {dates[0]} to {dates[-1]} -- {stats['count']} days")
	print("Daily max (C):", [round(x, 2) for x in tmax])
	print("Daily min (C):", [round(x, 2) for x in tmin])
	print("Highest Temperature (C):", stats["highest_c"])
	print("Lowest Temperature (C):", stats["lowest_c"])
	print("Average Temperature (C):", stats["average_c"])
	print("Temperatures in Fahrenheit:", np.round(stats["fahrenheit"], 2).tolist())
	print("Number of days with Temperature above 20C:", stats["days_above_20c"])

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

