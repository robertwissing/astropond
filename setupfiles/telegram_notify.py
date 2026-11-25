"""
telegram_notify.py

Simple Telegram notification library + CLI.

Environment variables:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Library:

    from telegram_notify import TelegramNotifier

    tg = TelegramNotifier()
    tg.send_message("Simulation started.")
    tg.send_photo("density.png", caption="Density")
    tg.send_video("kh_movie.mp4", caption="KH evolution")
    tg.send_file("output.tar.gz", caption="Checkpoint")
    tg.send("anything.ext", caption="dispatched by extension")
    tg.send_album(["a.png", "b.png", "c.png"], caption="Panels")

CLI:

    python telegram_notify.py fig1.png fig2.pdf -c "sedov N=64"
    python telegram_notify.py movie.mp4
    python telegram_notify.py *.png --album -c "all panels"
    python telegram_notify.py -m "run finished"
    python telegram_notify.py report.png --as-document   # full-res original
"""

from pathlib import Path
import argparse
import mimetypes
import os
import sys
import time

import requests

# Telegram API limits
CAPTION_MAX = 1024
MESSAGE_MAX = 4096
PHOTO_MAX_BYTES = 10 * 1024 * 1024   # sendPhoto limit; larger images go as documents
FILE_MAX_BYTES = 50 * 1024 * 1024    # bot API upload limit
ALBUM_MAX = 10

PHOTO_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
ANIMATION_EXTS = {".gif"}


def _truncate(text, limit):
    if text and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None, timeout=60, retries=3):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        self.retries = retries

        if not self.token:
            raise ValueError(
                "Telegram bot token not provided.\n"
                "Set TELEGRAM_BOT_TOKEN or pass token=..."
            )

        if not self.chat_id:
            raise ValueError(
                "Telegram chat ID not provided.\n"
                "Set TELEGRAM_CHAT_ID or pass chat_id=..."
            )

        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def _post(self, endpoint, data=None, files=None):
        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/{endpoint}",
                    data=data,
                    files=files,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2 ** attempt)
                self._rewind(files)
                continue

            if response.status_code == 429:
                retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                time.sleep(retry_after + 1)
                self._rewind(files)
                continue

            if response.status_code >= 500:
                last_error = RuntimeError(f"Telegram server error {response.status_code}")
                time.sleep(2 ** attempt)
                self._rewind(files)
                continue

            result = response.json()
            if not result.get("ok", False):
                raise RuntimeError(f"Telegram API error: {result}")
            return result

        # requests includes the full request URL in connection exceptions.  A
        # Telegram Bot API URL contains the bot token, so never propagate that
        # exception text without redacting the credential first.
        error_detail = str(last_error)
        if self.token:
            error_detail = error_detail.replace(self.token, "<redacted>")
        raise RuntimeError(
            f"Telegram request failed after {self.retries} attempts: "
            f"{error_detail}"
        )

    @staticmethod
    def _rewind(files):
        # file handles must be rewound before a retry re-reads them
        if files:
            for f in files.values():
                if hasattr(f, "seek"):
                    f.seek(0)

    @staticmethod
    def _check_file(filename):
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(f"No such file: {filename}")
        size = filename.stat().st_size
        if size == 0:
            raise ValueError(f"Refusing to send empty file: {filename}")
        if size > FILE_MAX_BYTES:
            raise ValueError(
                f"{filename} is {size / 1e6:.0f} MB; the bot API upload limit is 50 MB"
            )
        return filename, size

    def send_message(self, text, parse_mode=None, silent=False):
        """
        Send a text message.

        Parameters
        ----------
        text : str
        parse_mode : "MarkdownV2", "HTML", or None
        silent : bool
            Deliver without a notification sound.
        """
        data = {
            "chat_id": self.chat_id,
            "text": _truncate(text, MESSAGE_MAX),
            "disable_notification": silent,
        }

        if parse_mode is not None:
            data["parse_mode"] = parse_mode

        return self._post("sendMessage", data=data)

    def _send_media(self, endpoint, field, filename, caption, silent):
        filename, _ = self._check_file(filename)
        with filename.open("rb") as f:
            return self._post(
                endpoint,
                data={
                    "chat_id": self.chat_id,
                    "caption": _truncate(caption or filename.name, CAPTION_MAX),
                    "disable_notification": silent,
                },
                files={field: f},
            )

    def send_photo(self, filename, caption=None, silent=False):
        """
        Send an image. Telegram recompresses photos and caps them at 10 MB;
        anything over the cap is sent as a document (original preserved).
        """
        filename, size = self._check_file(filename)
        if size > PHOTO_MAX_BYTES:
            return self.send_file(filename, caption=caption, silent=silent)
        return self._send_media("sendPhoto", "photo", filename, caption, silent)

    def send_video(self, filename, caption=None, silent=False):
        """Send a video (mp4 plays inline in the chat)."""
        return self._send_media("sendVideo", "video", filename, caption, silent)

    def send_animation(self, filename, caption=None, silent=False):
        """Send a GIF/animation."""
        return self._send_media("sendAnimation", "animation", filename, caption, silent)

    def send_file(self, filename, caption=None, silent=False):
        """Send any file as a document (no recompression, original filename)."""
        return self._send_media("sendDocument", "document", filename, caption, silent)

    def send(self, filename, caption=None, silent=False, as_document=False):
        """
        Send a file, dispatching on extension:
        png/jpg/webp -> photo, mp4/mov -> video, gif -> animation,
        everything else (pdf, txt, tar.gz, ...) -> document.
        """
        filename = Path(filename)
        ext = filename.suffix.lower()
        if as_document:
            return self.send_file(filename, caption=caption, silent=silent)
        if ext in PHOTO_EXTS:
            return self.send_photo(filename, caption=caption, silent=silent)
        if ext in VIDEO_EXTS:
            return self.send_video(filename, caption=caption, silent=silent)
        if ext in ANIMATION_EXTS:
            return self.send_animation(filename, caption=caption, silent=silent)
        return self.send_file(filename, caption=caption, silent=silent)

    def send_album(self, filenames, caption=None, silent=False):
        """
        Send up to 10 photos/videos as a single grouped album.
        The caption is attached to the first item. Non-media files are rejected.
        """
        import json

        filenames = [self._check_file(f)[0] for f in filenames]
        if not 1 <= len(filenames) <= ALBUM_MAX:
            raise ValueError(f"Album must contain 1-{ALBUM_MAX} files, got {len(filenames)}")

        media, files = [], {}
        for i, filename in enumerate(filenames):
            ext = filename.suffix.lower()
            if ext in PHOTO_EXTS:
                mtype = "photo"
            elif ext in VIDEO_EXTS:
                mtype = "video"
            else:
                raise ValueError(f"Albums only take photos/videos, not {filename.name}")
            key = f"file{i}"
            item = {"type": mtype, "media": f"attach://{key}"}
            if i == 0 and caption:
                item["caption"] = _truncate(caption, CAPTION_MAX)
            media.append(item)
            mime = mimetypes.guess_type(str(filename))[0] or "application/octet-stream"
            files[key] = (filename.name, filename.open("rb"), mime)

        try:
            return self._post(
                "sendMediaGroup",
                data={
                    "chat_id": self.chat_id,
                    "media": json.dumps(media),
                    "disable_notification": silent,
                },
                files=files,
            )
        finally:
            for f in files.values():
                f[1].close()


class _NullNotifier:
    """No-op stand-in used when TELEGRAM_BOT_TOKEN is unset.

    Any ``tg.send_*(...)`` call becomes a silent no-op so analysis scripts run
    without a Telegram token configured (e.g. off the cluster).
    """

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def make_notifier(token=None, chat_id=None, **kwargs):
    """Return a TelegramNotifier if a token is configured, else a no-op stub.

    Avoids constructing (and importing the config-requiring) TelegramNotifier
    when TELEGRAM_BOT_TOKEN is not set, so callers can keep a module-level
    ``tg = make_notifier()`` without crashing where notifications are unused.
    """
    if token or os.getenv("TELEGRAM_BOT_TOKEN"):
        return TelegramNotifier(token=token, chat_id=chat_id, **kwargs)
    return _NullNotifier()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Send files and/or a message to Telegram.",
        epilog="Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment.",
    )
    parser.add_argument("files", nargs="*", help="files to send (type inferred from extension)")
    parser.add_argument("-c", "--caption", help="caption for the file(s)")
    parser.add_argument("-m", "--message", help="text message to send (before any files)")
    parser.add_argument("--album", action="store_true",
                        help="group images/videos into albums of up to 10")
    parser.add_argument("--as-document", action="store_true",
                        help="send images uncompressed as documents")
    parser.add_argument("--silent", action="store_true", help="no notification sound")
    args = parser.parse_args(argv)

    if not args.files and not args.message:
        parser.error("nothing to send: give files and/or -m MESSAGE")

    tg = TelegramNotifier()

    if args.message:
        tg.send_message(args.message, silent=args.silent)
        print("Sent message")

    if args.album and args.files:
        for i in range(0, len(args.files), ALBUM_MAX):
            batch = args.files[i:i + ALBUM_MAX]
            tg.send_album(batch, caption=args.caption if i == 0 else None, silent=args.silent)
            print(f"Sent album ({len(batch)} files)")
    else:
        for filename in args.files:
            tg.send(filename, caption=args.caption, silent=args.silent,
                    as_document=args.as_document)
            print(f"Sent {filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
