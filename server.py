"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import InputRequiredResult

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

mcp = FastMCP("sf-311")
logger = logging.getLogger("sf311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}


def _check_name(kind: str, value: str | None, choices: list[str]) -> None:
    """Raise a ToolError that suggests real names when `value` is not one of `choices`."""
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f"Did you mean: {', '.join(suggestions)}?" if suggestions else "Call dataset_info to see the valid names."
    raise ToolError(f"Unknown {kind} {value!r}. {hint}")


def _masked(request: Request) -> dict:
    """Return the request as a dict with its address reduced to the block."""
    row = request.as_dict()
    match = HOUSE_NUMBER.match(row["address"])
    if match:
        row["address"] = f"{int(match.group(1)) // 100 * 100} block of {match.group(2).strip()}"
    return row


def _matching(service: str | None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name("service", service, SERVICES)
    _check_name("neighborhood", neighborhood, NEIGHBORHOODS)
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests, newest first, up to `limit` (1 to 50, default 20).

    Each row has the id, opened and closed times, status, service, subtype,
    agency, neighborhood, address, and source. All filters are optional, and
    names must match dataset_info exactly.

    Use this when you need to see example requests or read their details.
    Use count_requests instead when the question is "how many" or "which has
    the most", since this returns at most 50 rows.
    Use median_close_hours instead when the question is about how long
    requests took to close.
    Use get_request instead when you already have a request id.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}. Use count_requests to count more than {MAX_ROWS} requests.")
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [_masked(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: Literal["neighborhood", "service", "agency", "source", "weekday", "status"] | None = None,
) -> int | dict[str, int]:
    """Count 311 requests opened this week, optionally filtered by service and neighborhood.

    Without `group_by` it returns one number. With `group_by` it returns a dict
    of group to count, largest first, at most 50 groups. Names must match
    dataset_info exactly.

    Use this when the question is "how many" or "which has the most".
    Use search_requests instead when you need to see the requests themselves.
    Use median_close_hours instead when the question is about how long
    requests took to close, not how many there were.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    pairs = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(pairs[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests_by_hour(
    service: str | None = None,
    neighborhood: str | None = None,
) -> dict[str, int]:
    """Count requests opened in each hour of the day (San Francisco local time), busiest hour first.

    Returns a dict of hour ("0" to "23", where "14" means 2:00 pm to 2:59 pm)
    to count, largest first, with every hour that had at least one request.
    Filters are optional, and names must match dataset_info exactly.

    Use this when the question is about time of day: "which hour had the most
    requests", "when do people report X".
    Use count_requests instead for totals or for groups other than hour, such
    as weekday. Use search_requests instead when you need individual requests.
    """
    counts = Counter(r.opened.hour for r in _matching(service, neighborhood))
    pairs = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {str(hour): count for hour, count in pairs}


@mcp.tool(annotations={"readOnlyHint": True})
def count_in_neighborhood(
    ctx: Context,
    name: str,
    boundary: str | None = None,
) -> int | str | InputRequiredResult:
    """Count 311 requests in a neighborhood, official or local, such as "Mission" or "Lower Nob Hill".

    Returns one number. A local name can overlap several official
    neighborhoods. Then this tool asks the user which part to count, offering
    each overlapped neighborhood and "All of <name>", and counts their choice.
    Pass `boundary` to skip the question: one of the overlapped official
    neighborhoods, or "all". If the user declines, it returns a message and
    counts nothing.

    Use this when the question names a neighborhood people use in everyday
    speech. Use count_requests instead for official names with a service
    filter or a group_by. Use search_requests instead to see the requests.
    """
    if name in NEIGHBORHOODS:
        return sum(1 for r in REQUESTS if r.neighborhood == name)
    if name not in LOCAL_NAMES:
        suggestions = closest(name, NEIGHBORHOODS + LOCAL_NAMES)
        hint = f"Did you mean: {', '.join(suggestions)}?" if suggestions else "Call dataset_info to see the valid names."
        raise ToolError(f"Unknown neighborhood {name!r}. {hint}")

    in_area = [r for r in REQUESTS if r.local_name == name]
    parts = sorted({r.neighborhood for r in in_area if r.neighborhood})
    all_option = f"All of {name}"

    if boundary is None and len(parts) > 1:
        choice = read_choice(ctx, "boundary")
        if choice is None:
            return ask_choice("boundary", f"{name} overlaps {len(parts)} official neighborhoods. Which part should be counted?", parts + [all_option])
        if not choice.accepted:
            return f"No neighborhood was chosen, so nothing was counted for {name}."
        boundary = choice.value

    if boundary is None or boundary.lower() in ("all", all_option.lower()):
        return len(in_area)
    if boundary not in parts:
        raise ToolError(f"{boundary!r} is not part of {name}. Try one of: {', '.join(parts)}, or 'all'.")
    return sum(1 for r in in_area if r.neighborhood == boundary)


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: Literal["neighborhood", "agency"] | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for requests that have been closed this week.

    Without `group_by` it returns one number (None if nothing is closed). With
    `group_by` it returns a dict of group to median hours, fastest first, and
    leaves out groups with fewer than 10 closed requests. Open requests are
    ignored. Service names must match dataset_info exactly.

    Use this when the question is "how fast" or "which closes fastest or
    slowest", by neighborhood or by agency.
    Use count_requests instead when the question is how many requests there
    were. Use search_requests instead when you need individual requests.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return one 311 request by its service request id, with all its fields.

    The address is shown only to the block. The caller is logged.

    Use this when you already have an id, for example from search_requests or
    from the user.
    Use search_requests instead when you only know the service, neighborhood
    or status. Use count_requests instead for totals.
    """
    request = BY_ID.get(request_id)
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find valid ids.")
    logger.info("get_request %s by %s", request_id, caller_email() or "local")
    return _masked(request)


if __name__ == "__main__":
    mcp.run()
