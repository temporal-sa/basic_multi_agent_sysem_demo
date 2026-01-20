"""Tool definitions for the multi-agent personal assistant demo.

These functions are registered with the shared ``mytools`` package so that
LLM activities can call them via the OpenAI / Gemini tool-calling APIs.

The design mirrors the LangChain multi-agent personal assistant example:

- Low-level helpers that simulate calendar and email integrations
- High-level tools (``schedule_event`` and ``manage_email``) that the
  "supervisor" agent can invoke using natural language.

All implementations are intentionally side-effect-free and return
human-readable summaries so that the demo is safe to run in any
environment. In a real system, you would replace the helper functions
with calls into actual calendar/email services.
"""

from __future__ import annotations

from typing import List

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from src.resources.mytools.decorators import tool


def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: List[str],
    location: str = "",
) -> str:
    """Create a calendar event.

    This helper mirrors the LangChain example and represents a call to an
    external calendar system (Google Calendar, Outlook, etc.).

    In this demo implementation we simply return a summary string instead
    of performing any real I/O, so the workflow remains safe and easy to
    run locally.
    """
    attendee_count = len(attendees)
    attendees_display = ", ".join(attendees) if attendees else "no attendees"
    location_display = f" at {location}" if location else ""
    return (
        f"Event created: {title}{location_display} "
        f"from {start_time} to {end_time} with {attendee_count} attendees "
        f"({attendees_display})."
    )


def send_email(
    to: List[str],
    subject: str,
    body: str,
    cc: List[str] | None = None,
) -> str:
    """Send an email via an email provider.

    In a production system this function would call a provider such as
    SendGrid or the Gmail API. For the purposes of this demo, we only
    construct and return a descriptive string.
    """
    cc = cc or []
    recipients = ", ".join(to) if to else "<no recipients>"
    cc_recipients = f" (cc: {', '.join(cc)})" if cc else ""
    preview = body.strip()
    if len(preview) > 80:
        preview = preview[:77] + "..."
    return (
        f"Email sent to {recipients}{cc_recipients} - Subject: {subject!r} - Preview: {preview!r}"
    )


def get_available_time_slots(
    attendees: List[str],
    date: str,
    duration_minutes: int,
) -> List[str]:
    """Return pseudo-available time slots for a given date.

    The LangChain sample delegates to real calendar APIs; here we simply
    return a fixed set of slots so that the LLM and tools have something
    deterministic to work with during demos.
    """
    _ = (attendees, duration_minutes)  # unused in the stub implementation
    return ["09:00", "14:00", "16:00"]


@tool
def schedule_event(request: str) -> str:
    """High-level tool for scheduling calendar events from natural language.

    This function is exposed to the LLM as a tool. It mirrors the
    ``schedule_event`` tool in the original LangChain example, but instead
    of spinning up a nested agent it executes a small amount of Python
    logic and returns a concise summary.

    The ``request`` parameter contains the user's natural language
    description of the meeting they would like to schedule. In a real
    implementation you would parse that text into structured fields and
    call calendar APIs. For this demo we synthesize a simple event using
    the helper functions above.
    """
    # NOTE: For the demo we do not attempt to fully parse natural language.
    # Instead, we construct a plausible event that clearly documents what
    # happened, while still letting the LLM practice tool-selection.
    title = "Team meeting"
    # In a production system these values would be derived from ``request``.
    date = "2024-01-15"
    attendees = ["team@example.com"]

    # Reuse the low-level helper to pretend we queried availability.
    available_slots = get_available_time_slots(
        attendees=attendees,
        date=date,
        duration_minutes=60,
    )
    chosen_slot = available_slots[0]

    start_time = f"{date}T{chosen_slot}:00"
    end_time = f"{date}T{int(chosen_slot.split(':')[0]) + 1:02d}:00"

    event_summary = create_calendar_event(
        title=title,
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
        location="Virtual",
    )

    return (
        f"{event_summary} The event was scheduled in response to: {request!r}."
        " In a real system, natural language parsing would tailor the time,"
        " attendees, and location more precisely."
    )


@tool
def manage_email(request: str) -> str:
    """High-level tool for composing and sending emails from natural language.

    This tool mirrors the ``manage_email`` tool in the LangChain example.
    It is designed to be called by a supervisor agent that decides when an
    email should be sent as part of a larger multi-step task.
    """
    # Basic, deterministic mapping from a free-form request into a stub
    # "outgoing email". A real application would perform entity extraction,
    # lookup real email addresses, and add richer templates.
    to = ["recipient@example.com"]
    subject = "Automated message from the multi-agent demo"
    body = (
        "The user asked the assistant to perform the following action:\n\n"
        f"{request}\n\n"
        "This is a demo email generated by the Temporal multi-agent workflow."
    )

    summary = send_email(to=to, subject=subject, body=body)

    return (
        f"{summary}\n\n"
        "In other words, a professional email was composed and "
        "sent to the recipient(s) above, using the text of your "
        "request as the core message."
    )


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Weather agent tool that fetches real current conditions.

    This implementation uses the Open-Meteo APIs:

    - Geocoding API to resolve a free-form ``location`` into latitude/longitude
    - Forecast API to fetch current weather for those coordinates

    No API key is required. All network I/O happens inside an activity
    (via ``tool_activity``), keeping workflows deterministic.
    """
    normalized_unit = unit.lower()
    if normalized_unit not in {"celsius", "fahrenheit"}:
        normalized_unit = "celsius"

    # Step 1: Geocode the location to lat/lon
    geo_params = urlencode(
        {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }
    )
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{geo_params}"

    try:
        with urlopen(geo_url, timeout=5) as resp:
            geo_payload = resp.read().decode("utf-8")
        geo_data = json.loads(geo_payload)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return f"Weather lookup failed for {location!r}: {exc}"

    results = geo_data.get("results") or []
    if not results:
        return f"Could not find coordinates for {location!r}."

    first = results[0]
    lat = first.get("latitude")
    lon = first.get("longitude")
    name = first.get("name") or location
    country = first.get("country") or ""

    if lat is None or lon is None:
        return f"Geocoding did not return usable coordinates for {location!r}."

    # Step 2: Fetch current weather for those coordinates
    temp_unit_param = "celsius" if normalized_unit == "celsius" else "fahrenheit"
    forecast_params = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "temperature_unit": temp_unit_param,
        }
    )
    forecast_url = f"https://api.open-meteo.com/v1/forecast?{forecast_params}"

    try:
        with urlopen(forecast_url, timeout=5) as resp:
            forecast_payload = resp.read().decode("utf-8")
        forecast_data = json.loads(forecast_payload)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return f"Weather API request failed for {location!r}: {exc}"

    current = forecast_data.get("current_weather") or {}
    temp_value = current.get("temperature")
    wind_speed = current.get("windspeed")

    if temp_value is None:
        return f"Weather data was unavailable for {location!r}."

    temp_suffix = "°C" if normalized_unit == "celsius" else "°F"
    wind_part = f" Windspeed ~{wind_speed} km/h." if wind_speed is not None else ""

    where = f"{name}, {country}".strip(", ")
    return (
        f"Current weather for {where}: approximately {temp_value}{temp_suffix}.{wind_part}"
        " Data sourced from the Open-Meteo APIs."
    )


@tool
def company_research(company: str) -> str:
    """High-level hook into the company research sub-agent.

    The actual long-running analysis is performed by the
    `company_research_agent.AgentLoopWorkflow` child workflow. The
    supervisor workflow intercepts calls to this tool name and starts
    the child workflow instead of executing this stub directly.

    This function exists primarily so the tool appears in the schema
    presented to the LLM; it should not be invoked directly by
    `tool_activity` in normal operation.
    """
    return (
        "Company research requested for "
        f"{company!r}. The full analysis will be performed by the "
        "company_research_agent child workflow orchestrated by the "
        "supervisor agent."
    )
