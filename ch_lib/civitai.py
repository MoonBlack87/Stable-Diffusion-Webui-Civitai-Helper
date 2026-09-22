""" -*- coding: UTF-8 -*-
handle msg between js and python side
"""

import io
import os
import re
from PIL import Image
from . import util
from . import model
from . import downloader

SUFFIX = ".civitai"

URLS = {
    "query": "https://civitai.com/api/v1/models?",
    "modelPage": "https://civitai.com/models/",
    "modelId": "https://civitai.com/api/v1/models/",
    "modelVersionId": "https://civitai.com/api/v1/model-versions/",
    "hash": "https://civitai.com/api/v1/model-versions/by-hash/"
}

MODEL_TYPES = {
    "Checkpoint": "ckp",
    "TextualInversion": "ti",
    "Hypernetwork": "hyper",
    "LORA": "lora",
    "LoCon": "lycoris",
    "DoRA": "lora",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "Detection": "detection"
}

MODEL_CATEGORIES = {
    "character",
    "style",
    "celebrity",
    "concept",
    "clothing",
    "base model",
    "poses",
    "background",
    "tool",
    "buildings",
    "vehicle",
    "objects",
    "animal",
    "action",
    "assets"
}

FILE_TYPES = [
    "Model", "Config", "VAE" # , "Training Data"
]

# Current public Civitai API uses the NsfwLevel bitmask from
# src/server/common/enums.ts.
NSFW_LEVELS = {
    "PG": 1,
    "PG13": 2,
    "R": 4,
    "X": 8,
    "XXX": 16,
    "Blocked": 32, # Probably not actually visible through the API without being logged in on model owner account?
}


def get_civitai_headers(accept=None):
    """Build headers for current Civitai API/CDN requests."""
    headers = {}
    if accept:
        headers["Accept"] = accept

    # Keep the legacy setting key for backward compatibility. The original
    # extension shipped this option with the "civiai" typo.
    api_key = util.get_opts("ch_civiai_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def civitai_get(civitai_url: str):
    """
    Gets JSON from Civitai.
    return: dict:json or None
    """

    util.printD(f"Requesting Civitai: {civitai_url}")

    success, response = downloader.request_get(
        civitai_url,
        headers=get_civitai_headers("application/json")
    )

    if not success:
        return None

    # Parse and close the streamed response. Public model/version endpoints are
    # still /api/v1 in current Civitai.
    try:
        with response:
            return response.json()
    except ValueError as e:
        util.printD(util.indented_msg(
            f"""
            Parse response json failed
            Error: {str(e)}
            Response: {response.text}
            """
        ))
        return None


def append_parent_model_metadata(content):
    """
    Some model metadata is stored in a "parent" context.
    When we're fething a model by its hash, we're getting
    the metadata for that model *file*, not the model entry
    on Civitai, which may contain multiple versions.

    This method gets the parent metadata and appends it to
    our model file metadata.

    return: model metadata with parent description, creator,
    and permissions appended.
    """
    util.printD("Fetching Parent Model Information")
    parent_model = get_model_info_by_id(content["modelId"])

    if not parent_model:
        # Archived models can give the model version information but 404 on the model metadata itself.
        parent_model = {}

    metadatas = [
        "description", "tags", "allowNoCredit",
        "allowCommercialUse", "allowDerivatives",
        "allowDifferentLicense"
    ]

    content["creator"] = parent_model.get("creator", "{}")

    model_metadata = content["model"]
    for metadata in metadatas:
        model_metadata[metadata] = parent_model.get(metadata, "")

    return content


def get_model_info_by_hash(model_hash: str):
    """
    use this sha256 to get model info from civitai's api

    return:
        model info dict if a model is found
        {} if civitai does not have the model
        None if an error occurs.
    """
    util.printD("Request model info from civitai")

    if not model_hash:
        util.printD("hash is empty")
        return None

    try:
        content = civitai_get(f'{URLS["hash"]}{model_hash}')
    except Exception as e:
        util.printD(f"Failed to get model info by hash: {model_hash}")
        util.printD(f"Error: {str(e)}")
        return None

    if not content:
        return None

    #util.printD(content)

    content = append_parent_model_metadata(content)

    return content


def get_model_info_by_id(model_id: str) -> dict:
    """
    Fetches model info by its model id.
    returns: dict:model_info
    """

    util.printD(f"Request model info from civitai: {model_id}")

    if not model_id:
        util.printD("model_id is empty")
        return False

    content = civitai_get(f'{URLS["modelId"]}{model_id}')

    return content


def get_version_info_by_version_id(version_id: str) -> dict:
    """
    Gets model version info from Civitai by version id
    return: dict:model_info
    """
    util.printD("Request version info from civitai")

    if not version_id:
        util.printD("version_id is empty")
        return None

    content = civitai_get(f'{URLS["modelVersionId"]}{version_id}')

    if content:
        content = append_parent_model_metadata(content)

    return content


def get_version_info_by_model_id(model_id: str) -> dict:
    """
    Fetches version info by model id.
    returns: dict:version_info
    """

    model_info = get_model_info_by_id(model_id)
    if not model_info:
        util.printD(f"Failed to get model info by id: {model_id}")
        return None

    # check content to get version id
    versions = model_info.get("modelVersions", [])
    if len(versions) == 0:
        util.printD("Found no model versions")
        return None

    def_version = versions[0]
    if not def_version:
        util.printD("default version is None")
        return None

    version_id = def_version.get("id", "")

    if not version_id:
        util.printD("Could not get valid version id")
        return None

    # get version info
    version_info = get_version_info_by_version_id(f"{version_id}")
    if not version_info:
        util.printD(f"Failed to get version info by version_id: {version_id}")
        return None

    return version_info


def load_model_info_by_search_term(model_type, search_term):
    """
    get model info file's content by model type and search_term
    parameter: model_type, search_term
    return: model_info
    """
    util.printD(f"Load model info of {search_term} in {model_type}")
    if model.folders.get(model_type, None) is None:
        util.printD(f"unknown model type: {model_type}")
        return None

    # search_term = f"{subfolderpath}{model name}{ext}"
    # And it always start with a / even when there is no sub folder
    base, _ = os.path.splitext(search_term)
    model_info_base = base
    if base[:1] == "/":
        model_info_base = base[1:]

    if model_type == "lora" and model.folders['lycoris']:
        model_folders = [model.folders[model_type], model.folders['lycoris']]
    else:
        model_folders = [model.folders[model_type]]

    for model_folder in model_folders:
        model_info_filename = f"{model_info_base}{SUFFIX}{model.CIVITAI_EXT}"
        model_info_filepath = os.path.join(model_folder, model_info_filename)

        found = os.path.isfile(model_info_filepath)

        if found:
            break

    if not found:
        util.printD(f"Can not find model info file: {model_info_filepath}")
        return None

    return model.load_model_info(model_info_filepath)


def get_model_names_by_type_and_filter(model_type: str, metadata_filter: dict) -> list:
    """
    get model file names by model type
    parameter: model_type - string
    parameter: filter - dict, which kind of model you need
    return: model name list
    """

    if model_type == "lora" and model.folders['lycoris']:
        model_folders = [model.folders[model_type], model.folders['lycoris']]
    else:
        model_folders = [model.folders[model_type]]

    # set metadata_filter
    # only get models don't have a civitai info file
    no_info_only = False
    empty_info_only = False

    if metadata_filter:
        no_info_only = metadata_filter.get("no_info_only", False)
        empty_info_only = metadata_filter.get("empty_info_only", False)

    # get information from filter
    # only get those model names don't have a civitai model info file
    model_names = []
    for model_folder in model_folders:
        for root, _, files in os.walk(model_folder, followlinks=True):
            for filename in files:
                if is_valid_file(root, filename, no_info_only, empty_info_only):
                    model_names.append(filename)

    return model_names


def is_valid_file(root, filename, no_info_only, empty_info_only):
    """
    Filters through model files to determine if they are
    valid targets for downloading new metadata.

    return: bool
    """
    item = os.path.join(root, filename)
    # check extension
    base, ext = os.path.splitext(item)
    if ext not in model.EXTS:
        return False

    # find a model
    info_file = f"{base}{SUFFIX}{model.CIVITAI_EXT}"

    # check filter
    if os.path.isfile(info_file):
        if no_info_only:
            return False

        if empty_info_only:
            # load model info
            model_info = model.load_model_info(info_file)
            # check content
            if model_info and not model_info.get("id", "") == "":
                # find a non-empty model info file
                return False

    return True


def get_model_names_by_input(model_type, empty_info_only):
    """ return: list of model filenames with empty civitai info files """
    return get_model_names_by_type_and_filter(model_type, {"empty_info_only": empty_info_only})


# get id from url
def get_model_id_from_url(url: str, include_model_ver=False):
    """
    Return the model id from a Civitai model URL or numeric model id.

    If include_model_ver is True, return a (model_id, model_version_id)
    tuple. model_version_id is None when it is not present in the URL.
    """
    util.printD("Run get_model_id_from_url")

    if not url:
        util.printD("url or model id can not be empty")
        return None

    value = str(url).strip()
    if not value:
        util.printD("url or model id can not be empty")
        return None

    if value.isnumeric():
        if include_model_ver:
            return (value, None)
        return value

    model_m = re.search(r"/models/(\d+)", value, re.IGNORECASE)
    if not model_m:
        util.printD("There is no model id in this url")
        return None

    model_id = model_m.group(1)

    ver_m = re.search(
        r"(?:[?&]|^)modelVersionId=(\d+)(?:&|$)",
        value,
        re.IGNORECASE
    )
    model_version_id = ver_m.group(1) if ver_m else None

    if not include_model_ver:
        return model_id

    return (model_id, model_version_id)


def preview_exists(model_path):
    """ Search for existing preview image. return True if it exists, else false """

    previews = model.get_potential_model_preview_files(model_path)

    for prev in previews:
        if os.path.isfile(prev):
            return True

    return False


def get_image_url(img_dict, max_size_preview):
    """
    Return the image delivery URL supplied by the current Civitai API.

    Current /api/v1 model and model-version responses already run image URLs
    through Civitai's edge URL builder with original=true. Older Helper code
    attempted to rewrite /width=N/ URL fragments, which no longer matches the
    current edge URL contract and can produce invalid URLs.
    """
    return img_dict.get("url")


def fetch_preview_image(img_dict, max_size_preview, nsfw_preview_threshold):
    """
    Fetch and decode one Civitai preview image without requiring Content-Length.

    Civitai's image CDN may use streaming/chunked responses, so the model-file
    downloader (which requires Content-Length for resumable downloads) must not
    be used for preview media.
    """
    img_url = img_dict.get("url")
    if not img_url:
        return (False, "Civitai image response did not contain a URL.")

    image_rating = img_dict.get("nsfwLevel", NSFW_LEVELS["Blocked"])
    if image_rating > 1:
        util.printD(f"This image is NSFW: {image_rating}")
        threshold = NSFW_LEVELS.get(nsfw_preview_threshold, NSFW_LEVELS["PG"])
        if threshold < image_rating:
            return (False, f"Skipped preview with NSFW level {image_rating}.")

    preview_type = img_dict.get("type")
    if preview_type != "image":
        return (False, f"Preview is not an image ({preview_type}).")

    img_url = get_image_url(img_dict, max_size_preview)
    if not img_url:
        return (False, "Could not resolve Civitai preview URL.")

    success, response_or_error = downloader.request_get(
        img_url,
        headers=get_civitai_headers("image/*")
    )
    if not success:
        return (False, str(response_or_error))

    response = response_or_error

    try:
        with response:
            content_type = response.headers.get("Content-Type", "").lower()
            if (
                content_type
                and not content_type.startswith("image/")
                and (
                    content_type.startswith("text/")
                    or "json" in content_type
                    or "html" in content_type
                )
            ):
                return (
                    False,
                    f"Civitai preview returned unexpected Content-Type: {content_type}"
                )

            payload = io.BytesIO()
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    payload.write(chunk)

        if payload.tell() == 0:
            return (False, "Civitai preview response was empty.")

        payload.seek(0)
        with Image.open(payload) as source:
            source.seek(0)
            source.load()

            bands = source.getbands()
            if "A" in bands:
                image = source.convert("RGBA")
            elif source.mode != "RGB":
                image = source.convert("RGB")
            else:
                image = source.copy()

        return (True, image)

    except (OSError, ValueError) as e:
        return (False, f"Could not decode Civitai preview image: {e}")


def save_preview_image(
    path,
    img_dict,
    max_size_preview,
    nsfw_preview_threshold
):
    """Fetch a Civitai image and atomically save it as a real PNG."""
    success, image_or_error = fetch_preview_image(
        img_dict,
        max_size_preview,
        nsfw_preview_threshold
    )
    if not success:
        return (False, image_or_error)

    tmp_path = f"{path}.downloading"
    try:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

        image_or_error.save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
        util.printD(f"Preview image saved to: {path}")
        return (True, path)

    except OSError as e:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return (False, f"Could not save preview image: {e}")


def verify_preview(path, img_dict, max_size_preview, nsfw_preview_threshold):
    """Download one valid Civitai preview image."""
    success, result = save_preview_image(
        path,
        img_dict,
        max_size_preview,
        nsfw_preview_threshold
    )

    if not success:
        util.printD(result)
        yield (False, result)
        return

    yield (True, result)


# get preview image by model path
# image will be saved to file, so no return
def get_preview_image_by_model_path(
    model_path: str,
    max_size_preview,
    nsfw_preview_threshold,
    preferred_preview=None,
    images=None,
    force=False
):
    """
    Downloads a preview image for a model if one doesn't already exist.
    Skips images that are more NSFW than the user's NSFW threshold
    """
    util.printD("Downloading model image.")

    if not model_path:
        util.printD("model_path is empty")
        return

    if not os.path.isfile(model_path):
        util.printD(f"model_path is not a file: {model_path}")
        return

    base, _ = os.path.splitext(model_path)
    preview_path = f"{base}.preview.png"  # TODO png not strictly required
    info_file = f"{base}{SUFFIX}{model.CIVITAI_EXT}"

    # need to download preview image
    util.printD(f"Checking preview image for model: {model_path}")

    if preview_exists(model_path) and not force:
        output = "Existing model image found. Skipping."
        util.printD(output)
        yield output
        return

    # Normally previews are loaded from the saved metadata file. The manual
    # Civitai URL workflow can pass the exact selected version's images
    # directly so preview creation never depends on stale metadata on disk.
    if images is None:
        if not os.path.isfile(info_file):
            return

        try:
            images = model.load_model_info(info_file)["images"]

        except (KeyError, TypeError):
            return

    if not images:
        util.printD(f"No preview images returned for model: {model_path}")
        yield "Civitai returned no preview images for the selected model version."
        return

    if preferred_preview:
        img_url = preferred_preview
        for img_dict in images:
            if img_dict["url"] == preferred_preview:
                img_url = get_image_url(img_dict, max_size_preview)
                break

        preferred_data = {"url": img_url, "type": "image", "nsfwLevel": 1}
        for img_dict in images:
            if img_dict.get("url") == preferred_preview:
                preferred_data = img_dict
                break

        success, msg = save_preview_image(
            preview_path,
            preferred_data,
            max_size_preview,
            nsfw_preview_threshold
        )

        if success:
            if force:
                for existing_preview in model.get_potential_model_preview_files(model_path):
                    if (
                        os.path.isfile(existing_preview)
                        and os.path.realpath(existing_preview) != os.path.realpath(preview_path)
                    ):
                        try:
                            os.remove(existing_preview)
                        except OSError as e:
                            util.printD(f"Could not remove old preview {existing_preview}: {e}")
            return

        util.printD(msg)
        util.printD("Failed to download preferred preview. Trying to find another")

    for img_dict in images:
        for result in verify_preview(
                preview_path, img_dict, max_size_preview, nsfw_preview_threshold
        ):
            if not isinstance(result, str):
                success, _ = result
                # Only download one image
                if success:
                    if force:
                        for existing_preview in model.get_potential_model_preview_files(model_path):
                            if (
                                os.path.isfile(existing_preview)
                                and os.path.realpath(existing_preview) != os.path.realpath(preview_path)
                            ):
                                try:
                                    os.remove(existing_preview)
                                except OSError as e:
                                    util.printD(f"Could not remove old preview {existing_preview}: {e}")
                    return

                break

            yield result

    util.printD(f"Could not find any valid preview images for model: {model_path}")
    yield


# search local model by version id in 1 folder, no subfolder
# return - model_info
def search_local_model_info_by_version_id(folder: str, model_ids: dict) -> dict:
    """ Searches a folder for model_info files,
        returns the model_info from a file if its id matches the model id.
    """
    util.printD("Searching local model by version id")
    util.printD(f"folder: {folder}")
    util.printD(f"model_ids: {model_ids}")

    version_id = model_ids["version"]
    model_id = model_ids["model"]

    if not folder:
        util.printD("folder is none")
        return None

    if not os.path.isdir(folder):
        util.printD("folder is not a dir")
        return None

    if not (version_id and model_id):
        util.printD("missing ID for model/version")
        return None

    # search civitai model info file
    for filename in os.listdir(folder):
        # check ext
        base, ext = os.path.splitext(filename)
        if ext == model.CIVITAI_EXT:
            # find info file
            if not (len(base) > 8 and base[-8:] == SUFFIX):
                # not a civitai info file
                continue

            # find a civitai info file
            path = os.path.join(folder, filename)

            try:
                model_info = model.load_model_info(path)
                existing_version_id = model_info.get("id", None)
                existing_model_id = model_info["modelId"]

            except Exception:
                continue

            # util.printD(f"Compare version id, src: {model_id}, target:{version_id}")
            if f"{existing_version_id}" == f"{version_id}":
                # find the one
                filepath = model.locate_model_from_partial(folder, base[:-8])
                return f"{filepath}"

    return None


def get_model_id_from_model_path(model_path: str):
    """ return model_id using model_path """
    # get model info file name
    base, _ = os.path.splitext(model_path)
    info_file = f"{base}{SUFFIX}{model.CIVITAI_EXT}"

    if not os.path.isfile(info_file):
        return None

    # get model info
    model_info_file = model.load_model_info(info_file)
    local_version_id = model_info_file.get("id", None)
    model_id = model_info_file.get("modelId", None)

    if None in [model_id, local_version_id]:
        return None

    return (model_id, local_version_id)


def check_model_new_version_by_path(model_path: str, delay: float = 0.2) -> tuple:
    """
    check new version for a model by model path
    return (
        model_path, model_id, model_name, new_verion_id,
        new_version_name, description, download_url, img_url
    )
    """

    if not (model_path and os.path.isfile(model_path)):
        util.printD(f"model_path is not a file: {model_path}")
        return None

    result = get_model_id_from_model_path(model_path)
    if not result:
        return None

    model_id, local_version_id = result

    # get model info by id from civitai
    model_info = get_model_info_by_id(model_id)

    util.delay(delay)

    if not model_info:
        return None

    model_versions = model_info.get("modelVersions", [])

    if len(model_versions) == 0:
        return None

    current_version = model_versions[0]
    if not current_version:
        return None

    current_version_id = current_version.get("id", False)

    util.printD(f"Compare version id, local: {local_version_id}, remote: {current_version_id}")

    if not (current_version_id and current_version_id != local_version_id):
        return None

    model_name = model_info.get("name", "")
    new_version_name = current_version.get("name", "")
    description = current_version.get("description", "")
    download_url = current_version.get("downloadUrl", "")

    # get 1 preview image
    try:
        img_url = current_version["images"][0]["url"]
    except (IndexError, KeyError):
        img_url = ""

    return (
        model_path, model_id, model_name, current_version_id,
        new_version_name, description, download_url, img_url
    )


def check_single_model_new_version(root, filename, model_type, delay):
    """
    return: True if a valid model has a new version.
    """
    # check ext
    item = os.path.join(root, filename)
    _, ext = os.path.splitext(item)

    if ext not in model.EXTS:
        return False

    # find a model
    request = check_model_new_version_by_path(item, delay)

    if not request:
        return False

    request = request + (model_type,)

    # model_path, model_id, model_name, version_id, new_version_name, description, downloadUrl, img_url = request
    model_ids = {
        'model': request[1],
        'version': request[3],
    }

    # check exist
    if not (model_ids['version'] and model_ids['model']):
        return False

    # search this new version id to check if this model is already downloaded
    target_model_info = search_local_model_info_by_version_id(root, model_ids)
    if target_model_info:
        util.printD("New version already exists")
        return False

    return request


def check_models_new_version_by_model_types(model_types: list, delay: float = 0.2) -> list:
    """
    check all models of model_types for new version
    parameter: delay - float, how many seconds to delay between each request to civitai
    return: new_versions
        a list for all new versions, each one is
        (model_path, model_id, model_name, new_verion_id,
        new_version_name, description, download_url, img_url)
    """
    util.printD("Checking models' new version")

    if not model_types:
        return []

    # check model types, which could be a string as 1 type
    mts = []
    if isinstance(model_types, str):
        mts.append(model_types)
    elif isinstance(model_types, list):
        mts = model_types
    else:
        util.printD("Unknown model types:")
        util.printD(model_types)
        return []

    # new version list
    new_versions = []
    new_version_ids = []

    # walk all models
    for model_type, model_folder in model.folders.items():
        if model_type not in mts:
            continue

        util.printD(f"Scanning path: {model_folder}")
        for root, _, files in os.walk(model_folder, followlinks=True):
            for filename in files:
                version = check_single_model_new_version(root, filename, model_type, delay)

                if not version:
                    continue

                # model_path, model_id, model_name, version_id, new_version_name, description, downloadUrl, img_url = version
                version_id = version[3]

                if version_id in new_version_ids:
                    continue

                # add to list
                new_versions.append(version)
                new_version_ids.append(version_id)

    return new_versions


def move_model_to_subfolder(filepath, model_info):
    model_id = model_info["modelId"]

    if model_id == "":
        return None

    content = civitai_get(f'{URLS["modelId"]}{model_id}')

    tags = content["tags"]

    # iterate through tags until we find one that matches MODEL_CATEGORIES

    for tag in tags:
        if tag in MODEL_CATEGORIES:
            model_category = tag

            # create subfolder if it doesn't exist
            # check to make sure the model is not already in the correct subfolder
            if model_category in filepath:
                return filepath

            # get the file path without the filename
            folderpath = os.path.dirname(filepath)

            # create the new folder path
            new_folder_path = os.path.join(folderpath, model_category)

            # create the new folder if it doesn't exist
            if not os.path.isdir(new_folder_path):
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                else:
                    console.log(f"`{new_folder_path}` is not a directory. Skipping. Please delete or move the file `{new_folder_path}`.")
                    break

            util.printD(f"Moving model from {filepath} to {new_folder_path}")

            # move the file to the new folder
            new_filepath = os.path.join(new_folder_path, os.path.basename(filepath))
            os.rename(filepath, new_filepath)

            return new_filepath

    util.printD("WARNING: Unable to find tag for folder")

    return filepath
