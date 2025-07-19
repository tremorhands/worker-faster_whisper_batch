import yt_dlp

def audio_from_url(url: str, path_to_save: str, is_playlist: bool = False) -> dict:
    """Download audio and return video info"""
    ydl_opts = {
        'format': 'bestaudio[ext=opus][abr<=96]/bestaudio[abr<=96]/bestaudio',
        'outtmpl': path_to_save,
        'postprocessors': [
            {
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            }
        ],
        'postprocessor_args': [
            '-ac', '1',
            '-ar', '16000',
            '-c:a', 'pcm_s16le'
        ],
        'prefer_ffmpeg': True,
        'external_downloader': 'aria2c',  
        'external_downloader_args': ['-x', '16'],    
        'noplaylist': not is_playlist
    }

    start = time.time()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # downloaded_file = ydl.prepare_filename(info)
        download_time = round(time.time() - start,1)
        # logging.info(f"Downloaded file: {downloaded_file} за {round(time.time() - start,1)}. Кодек: {info.get("acodec")}, Битрейт: {info.get("abr")}, Размер {round(info.get("filesize", 0) / 1024 / 1024, 2), "МБ"}")

    result = {
        'duration': info.get("duration", "Unknown Duration"),
        'title': info.get("title", "Unknown Title"),
        'channel': info.get("uploader", "Unknown Channel"),
        'entries': info.get("entries", []),
        'download_time': download_time
    }
 
    return result