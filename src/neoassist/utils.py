import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def _extract_zip(path: str, local_dir: str) -> None:
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(local_dir)


def _extract_tar(path: str, local_dir: str) -> None:
    with tarfile.open(path, "r") as tar_ref:
        tar_ref.extractall(local_dir)


def _extract_tgz(path: str, local_dir: str) -> None:
    with tarfile.open(path, "r:gz") as tar_ref:
        tar_ref.extractall(local_dir)


def _extract_tar_gz(path: str, local_dir: str) -> None:
    with tarfile.open(path, "r:gz") as tar_ref:
        tar_ref.extractall(local_dir)


SUPPORTED_ARCHIVES: dict[str | tuple[str], Callable] = {
    ".zip": _extract_zip,
    ".tar": _extract_tar,
    ".tgz": _extract_tgz,
    (".tar", ".gz"): _extract_tar_gz,
}


def is_archive(path: str) -> bool:
    """Возвращает True, если файл является архивом, иначе False."""
    path = Path(path)
    return path.suffix in SUPPORTED_ARCHIVES or tuple(path.suffixes[-2:]) in SUPPORTED_ARCHIVES


def unpack_archive(path: str, local_dir: str) -> None:
    """Распаковывает архив.

    Args:
        path: Путь к архиву.
        local_dir: Путь к директории, куда должен быть распакован архив.
    """
    path = Path(path)
    handler = None

    if path.suffix in SUPPORTED_ARCHIVES:
        handler = SUPPORTED_ARCHIVES[path.suffix]
    elif tuple(path.suffixes[-2:]) in SUPPORTED_ARCHIVES:
        handler = SUPPORTED_ARCHIVES[tuple(path.suffixes[-2:])]

    if handler:
        handler(path, local_dir)
    else:
        raise ValueError(f"Неизвестный формат архива: {path}")


def download_hf_dataset(repo_id: str, local_dir: str, token: str, filename: str | None = None) -> str:
    """Скачивает датасет с Hugging Face Hub и возвращает путь к локальной директории с ним.

    Функция поддерживает два режима загрузки:
    - Если `filename` не указан — скачивается весь датасет с помощью `snapshot_download`.
    - Если указан `filename` — скачивается только указанный архив,
      который распаковывается в `local_dir`.

    Args:
        repo_id: Идентификатор репозитория на HuggingFace в формате 'user/repo'.
        local_dir: Локальный путь, куда будет сохранен датасет.
        token: Токен аутентификации для доступа к приватным репозиториям.
        filename: Имя архива для скачивания, содержащий датасет.

    Returns:
        str: Абсолютный путь к локальной директории, содержащей датасет.
    """
    local_dir = Path(local_dir).resolve()
    if filename is None:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token, repo_type="dataset")
    else:
        downloaded_path = Path(
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, token=token, repo_type="dataset")
        ).resolve()

        if is_archive(downloaded_path):
            unpack_archive(downloaded_path, local_dir)

    return str(local_dir)


def abs_path(relative: str | Path, base_file: str | Path) -> Path:
    """Возвращает абсолютный путь, независимый от текущей рабочей директории."""
    return (Path(base_file).parent / relative).resolve()
