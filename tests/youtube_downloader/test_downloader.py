from piano_partner.youtube_downloader.downloader import _format_selector


def test_format_selector_no_cap():
    assert _format_selector(None) == (
        "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
        "best[ext=mp4]/best"
    )


def test_format_selector_1080():
    assert _format_selector(1080) == (
        "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
        "best[height<=1080][ext=mp4]/best[height<=1080]"
    )


def test_format_selector_720():
    assert _format_selector(720) == (
        "bestvideo[height<=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
        "best[height<=720][ext=mp4]/best[height<=720]"
    )
