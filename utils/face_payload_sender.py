from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import requests
from requests import HTTPError


@dataclass
class FacePayload:
    faceVector: list[float] = field(default_factory=list)
    age: float = 0.0
    gender: str = ""
    # confidence: float | None = None
    # locker_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # if data["confidence"] is None:
        #     data.pop("confidence")
        # if data["locker_id"] is None:
        #     data.pop("locker_id")
        return data


def _to_float_list(faceVector: Sequence[Any]) -> list[float]:
    return [float(value) for value in faceVector]


def build_face_payload(
    faceVector: Sequence[Any],
    age: float,
    gender: str,
    # confidence: float | None = None,
    # locker_id: int | None = None,
) -> FacePayload:
    return FacePayload(
        faceVector=_to_float_list(faceVector),
        age=float(age),
        gender=str(gender),
        # confidence=None if confidence is None else float(confidence),
        # locker_id=locker_id,
    )


def print_face_payload(payload: FacePayload | Mapping[str, Any]) -> None:
    data = payload.to_dict() if isinstance(payload, FacePayload) else dict(payload)
    print("\n[FACE PAYLOAD]")
    print(data)


def send_face_payload(
    url: str,
    payload: FacePayload | Mapping[str, Any],
    timeout: int | float = 10,
    headers: Mapping[str, str] | None = None,
) -> requests.Response:
    data = payload.to_dict() if isinstance(payload, FacePayload) else dict(payload)
    print_face_payload(data)
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json", **(dict(headers) if headers else {})},
            timeout=timeout,
        )
        response.raise_for_status()
        return response
    except HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        response_text = exc.response.text if exc.response is not None else ""
        print(f"[HTTP ERROR] POST {url} -> {status_code}")
        if response_text:
            print(f"[HTTP ERROR BODY] {response_text}")
        raise


def send_face_data(
    faceVector: Sequence[Any],
    age: float,
    gender: str,
    confidence: float | None = None,
    locker_id: int | None = None,
    timeout: int | float = 10,
    headers: Mapping[str, str] | None = None,
) -> requests.Response:
    payload = build_face_payload(
        faceVector=faceVector,
        age=round(float(age), 0),
        gender='male' if str(gender).lower() == 'nam' else 'female',
        # confidence=confidence,
        # locker_id=locker_id,
    )
    return send_face_payload('http://localhost:3000/api/v1/sessions/cico/face', payload, timeout=timeout, headers=headers)


if __name__ == "__main__":
    raise SystemExit(
        "Import utils.face_payload_sender and call send_face_data(url, faceVector, age, gender)."
    )