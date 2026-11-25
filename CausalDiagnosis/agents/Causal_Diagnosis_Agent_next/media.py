"""Only images returned by answered tool calls enter a later model request."""

import base64
import hashlib


def served_images(board):
    if board is None:
        return []
    images, seen = [], set()
    for call in board.calls:
        if not call["answered"]:
            continue
        result = call["answer"].get("result")
        attachments = result.get("attachments", []) if isinstance(result, dict) else []
        for item in attachments if isinstance(attachments, list) else ():
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            mime, data = item.get("mime_type"), item.get("data")
            if mime not in ("image/png", "image/jpeg", "image/gif", "image/webp") or not isinstance(data, str):
                raise ValueError("a served image needs a supported media type and base64 data")
            try:
                raw = base64.b64decode(data, validate=True)
            except ValueError as error:
                raise ValueError("a served image has invalid base64 data") from error
            if not raw:
                raise ValueError("a served image must not be empty")
            identity = hashlib.sha256(raw).hexdigest()
            if (mime, identity) not in seen:
                seen.add((mime, identity))
                observations = call["answer"].get("observations", [])
                images.append({"observation_id": observations[0]["id"] if observations else None,
                               "mime_type": mime, "data": data, "sha256": identity})
    return images


def text_view(value):
    """Render attachment descriptors, preserving full bytes in recorded calls."""
    if isinstance(value, dict):
        return {key: text_view(item) for key, item in value.items() if not (value.get("type") == "image" and key == "data")}
    if isinstance(value, list):
        return [text_view(item) for item in value]
    return value
