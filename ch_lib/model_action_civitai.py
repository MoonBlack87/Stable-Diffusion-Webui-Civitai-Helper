""" -*- coding: UTF-8 -*-
handle msg between js and python side
"""
import os
import time
import re
import gradio as gr
from modules import sd_models
from . import util
from . import model
from . import civitai
from . import downloader
from . import templates


def get_metadata_skeleton():
    """
    Used to generate at least something when model is not on civitai.
    """
    metadata = {
        "id": "",
        "modelId": "",
        "name": "",
        "trainedWords": [],
        "baseModel": "Unknown",
        "description": "",
        "model": {
            "name": "",
            "type": "",
            "nsfw": "",
            "poi": ""
        },
        "files": [
            {
                "name": "",
                "sizeKB": 0,
                "type": "Model",
                "hashes": {
                    "AutoV2": "",
                    "SHA256": ""
                }
            }
        ],
        "tags": [],
        "downloadUrl": "",
        "skeleton_file": True
    }

    return metadata


def scan_single_model(filepath, model_type, refetch_old, organize_models, delay):
    """
    Gets model info for a model by feeding its sha256 hash into civitai's api

    return: success:bool
    """

    filename = os.path.basename(filepath)

    # find a model, get info file
    info_file, sd15_file = model.get_model_info_paths(filepath)

    output = ""

    use_auto_v3 = util.get_opts("ch_autov3")

    # check info file
    if model.metadata_needed(info_file, sd15_file, refetch_old):
        output = f"Creating model info for: {filename}"
        util.printD(output)
        yield output

        # get model's sha256
        result = None
        for result in util.gen_file_sha256(filepath, use_addnet_hash=use_auto_v3):
            if isinstance(result, tuple):
                yield result

        sha256_hash = result

        util.printD(f"model action sha256: {sha256_hash}")

        if not sha256_hash:
            output = f"failed generating SHA256 for model: {filename}"
            util.printD(output)
            yield output
            time.sleep(delay)
            yield False

        civitai_hash = sha256_hash
        if use_auto_v3:
            civitai_hash = sha256_hash[:12]

        yield "Requesting model information from Civitai"
        # use this sha256 to get model info from civitai
        model_info = civitai.get_model_info_by_hash(civitai_hash)

        if not model_info:
            model_info = dummy_model_info(filepath, civitai_hash, model_type)
            yield True

        # if model is lora and not already in a subfolder, move into subfolder based on its type (character,
        # clothing, etc.)
        if organize_models and model_type in ["lora", "lycoris"]:
            filepath = civitai.move_model_to_subfolder(filepath, model_info)

        model.process_model_info(filepath, model_info, model_type, refetch_old=refetch_old)

        # delay before next request, to prevent being treated as a DDoS attack
        time.sleep(delay)

    else:
        util.printD(f"Model metadata not needed for {filename}")

    yield True


def scan_model(scan_model_types, refetch_old, organize_models=False, progress=gr.Progress()):
    """ Scan model to generate SHA256, then use this SHA256 to get model info from civitai
        return output msg
    """

    delay = 0.2

    util.printD("Start scan_model")
    output = ""

    nsfw_preview_threshold = util.get_opts("ch_nsfw_threshold")

    max_size_preview = util.get_opts("ch_max_size_preview")

    # check model types
    if not scan_model_types:
        output = "Model Types is None, can not scan."
        util.printD(output)
        yield output
        return

    model_types = scan_model_types
    if isinstance(scan_model_types, str):
        # check if type is a string
        model_types = [scan_model_types]

    models = []
    for model_type, model_folder in model.folders.items():
        if model_type not in model_types:
            continue

        util.printD(f"Scanning path: {model_folder}")
        for root, _, files in os.walk(model_folder, followlinks=True):
            for filename in files:

                # check ext
                filepath = os.path.join(root, filename)
                _, ext = os.path.splitext(filepath)
                if ext not in model.EXTS:
                    continue

                models.append((filepath, model_type))

    count = [0, 0]
    total = len(models)
    for filepath, model_type in models:
        success = None

        tracker = (count[0], total)

        progress(
            tracker,
            desc="Scanning...",
            unit="models"
        )

        count[0] = count[0] + 1

        for result in scan_single_model(filepath, model_type, refetch_old, organize_models, delay):
            if isinstance(result, str):
                progress(tracker, desc=result, unit="models")
                continue

            if isinstance(result, tuple):
                percent, status = result
                progress(percent, desc=status)
                continue

            success = result
            break

        if not success:
            continue

        # set model_count
        count[1] = count[1] + 1

        # check preview image
        for _ in civitai.get_preview_image_by_model_path(
            filepath,
            max_size_preview,
            nsfw_preview_threshold
        ):
            pass

    # this previously had an image count, but it always matched the model count.
    output = f"Done. Successfully scanned {count[1]} of {len(models)} models."

    util.printD(output)

    yield output


def dummy_model_info(path, sha256_hash, model_type):
    """
    Fills model metadata with information we can get locally.
    """
    if not sha256_hash:
        return {}

    model_info = get_metadata_skeleton()

    autov2 = sha256_hash[:10]
    filename = os.path.basename(path)
    filesize = os.path.getsize(path) // 1024

    model_metadata = model_info["model"]
    file_metadata = model_info["files"][0]

    model_metadata["name"] = filename
    model_metadata["type"] = model_type

    file_metadata["name"] = filename
    file_metadata["sizeKB"] = filesize
    file_metadata["hashes"]["SHA256"] = sha256_hash
    file_metadata["hashes"]["AutoV2"] = autov2

    # We can't get data on the model from civitai, but some models
    # do store their training data.
    trained_words = model_info["trainedWords"]
    tags = model_info["tags"]

    try:
        read_metadata = sd_models.read_metadata_from_safetensors(path)
    except AssertionError:
        # model is not a safetensors file. This is fine,
        # it just doesn't have metadata we can read
        return model_info

    tag_frequency = read_metadata.get("ss_tag_frequency", {})

    prefix_re = re.compile(r"^\d+_")

    if isinstance(tag_frequency, dict):
        for trained_word in tag_frequency.keys():
            # kohya training scripts use
            # `{iterations}_{trained_word}`
            # for training finetune concepts.
            word = prefix_re.sub("", trained_word)
            trained_words.append(word)

            # "tags" in this case are just words used in image captions
            # when training the finetune model.
            # They may or may not be useful for prompting
            for tag in tag_frequency[trained_word].keys():
                tag = tag.replace(",", "").strip()
                if tag == "" or tag in tags:
                    continue
                tags.append(tag)

    elif isinstance(tag_frequency, str):
        word = prefix_re.sub("", trained_word)
        trained_words.append(word)

    return model_info


def _normalize_civitai_id(value):
    """Return a trimmed Civitai id string."""
    if value is None:
        return ""
    return str(value).strip()


def _resolve_civitai_input_ids(model_url_or_id, model_id="", model_version_id=""):
    """
    Resolve editable model/version id fields.

    When either editable id field contains a value, those fields are treated as
    authoritative. This lets the user change or clear a version id after the
    initial URL preview without the URL silently overriding the edit.
    """
    model_id = _normalize_civitai_id(model_id)
    model_version_id = _normalize_civitai_id(model_version_id)

    if not model_id and not model_version_id:
        parsed = civitai.get_model_id_from_url(
            model_url_or_id,
            include_model_ver=True
        )
        if not parsed:
            return (None, None, f"Failed to parse Civitai model id from: {model_url_or_id}")

        model_id, model_version_id = parsed
        model_id = _normalize_civitai_id(model_id)
        model_version_id = _normalize_civitai_id(model_version_id)

    if model_id and not model_id.isnumeric():
        return (None, None, f"Invalid Civitai model id: {model_id}")

    if model_version_id and not model_version_id.isnumeric():
        return (None, None, f"Invalid Civitai model version id: {model_version_id}")

    if not model_id and not model_version_id:
        return (None, None, "A model id or model version id is required.")

    return (model_id, model_version_id, None)


def _get_exact_version_info(model_id, model_version_id):
    """
    Fetch exactly the requested model version and verify that it belongs to the
    requested parent model.

    If no version id is provided, Civitai's first listed version is selected
    explicitly and returned to the UI for confirmation before anything is
    written.
    """
    fallback_version = False

    if model_version_id:
        version_info = civitai.get_version_info_by_version_id(model_version_id)
        if not version_info:
            return (None, None, None, f"Could not retrieve model version {model_version_id}.", False)

        actual_model_id = _normalize_civitai_id(version_info.get("modelId"))
        if not actual_model_id:
            return (None, None, None, "Civitai response did not include a parent model id.", False)

        if model_id and actual_model_id != model_id:
            return (
                None,
                None,
                None,
                (
                    f"Model/version mismatch: version {model_version_id} belongs to "
                    f"model {actual_model_id}, not model {model_id}."
                ),
                False
            )

        return (
            actual_model_id,
            _normalize_civitai_id(version_info.get("id")) or model_version_id,
            version_info,
            None,
            False
        )

    if not model_id:
        return (None, None, None, "A model id is required when no version id is supplied.", False)

    version_info = civitai.get_version_info_by_model_id(model_id)
    if not version_info:
        return (None, None, None, f"Could not retrieve a version for model {model_id}.", False)

    selected_version_id = _normalize_civitai_id(version_info.get("id"))
    if not selected_version_id:
        return (None, None, None, "Civitai response did not include a model version id.", False)

    fallback_version = True
    return (model_id, selected_version_id, version_info, None, fallback_version)


def _model_types_compatible(local_type, civitai_type):
    """LoRA and LyCORIS are compatible storage targets for this workflow."""
    if not civitai_type:
        return True
    if local_type == civitai_type:
        return True
    return {local_type, civitai_type}.issubset({"lora", "lycoris"})


def _format_match_preview(
    model_path,
    local_model_type,
    model_id,
    model_version_id,
    version_info,
    fallback_version
):
    """Build a compact, human-readable preview of the exact Civitai match."""
    parent = version_info.get("model", {}) or {}
    civitai_type_name = parent.get("type", "Unknown")
    civitai_local_type = civitai.MODEL_TYPES.get(civitai_type_name)

    remote_name = parent.get("name", "Unknown")
    version_name = version_info.get("name", "Unknown")
    base_model = version_info.get("baseModel", "Unknown")

    model_files = [
        file_info.get("name", "")
        for file_info in version_info.get("files", [])
        if file_info.get("type") == "Model"
    ]
    model_files = [name for name in model_files if name]
    file_text = ", ".join(model_files) if model_files else "Unknown"

    trained_words = version_info.get("trainedWords", []) or []
    trained_words_text = ", ".join(trained_words[:10]) if trained_words else "None"
    if len(trained_words) > 10:
        trained_words_text += ", ..."

    warning_lines = []
    if fallback_version:
        warning_lines.append(
            "**Warning:** No modelVersionId was supplied. The first version "
            "returned by Civitai was selected. Review the version id before writing."
        )

    if not _model_types_compatible(local_model_type, civitai_local_type):
        warning_lines.append(
            f"**Warning:** Local type is `{local_model_type}`, but Civitai maps "
            f"this model to `{civitai_local_type}` ({civitai_type_name})."
        )

    lines = [
        "### Civitai match preview",
        "",
        f"- Local file: `{model_path}`",
        f"- Civitai model: **{remote_name}** — modelId `{model_id}`",
        f"- Version: **{version_name}** — modelVersionId `{model_version_id}`",
        f"- Civitai type: `{civitai_type_name}`",
        f"- Base model: `{base_model}`",
        f"- Model file(s): `{file_text}`",
        f"- Trained words: {trained_words_text}",
        "",
        "**Nothing has been written yet.** Review or edit the ids, click "
        "**Get Model Info from Civitai** again if you changed them, then use "
        "**Write Selected Model Info**."
    ]

    if warning_lines:
        lines.extend([""] + warning_lines)

    return "\n".join(lines)


def _get_version_preview_images(version_info):
    """Return allowed preview image URLs for the exact selected version."""
    images = []
    nsfw_preview_threshold = util.get_opts("ch_nsfw_threshold")
    max_size_preview = util.get_opts("ch_max_size_preview")

    for image in version_info.get("images", []) or []:
        if image.get("type") != "image":
            continue

        rating = image.get("nsfwLevel", 32)
        if civitai.NSFW_LEVELS[nsfw_preview_threshold] < rating:
            continue

        url = image.get("url")
        if not url:
            continue

        try:
            url = civitai.get_image_url(image, max_size_preview)
        except (KeyError, TypeError):
            pass

        images.append(url)

    return images


def preview_model_info_by_input(
    model_type,
    model_name,
    model_url_or_id,
    model_id="",
    model_version_id=""
):
    """
    Resolve a local model against Civitai without writing files.

    Returns a confirmation state plus editable model/version ids and a preview.
    """
    model_path = model.get_model_path_by_type_and_name(model_type, model_name)
    if model_path is None:
        return (
            {},
            _normalize_civitai_id(model_id),
            _normalize_civitai_id(model_version_id),
            "Could not get local model path.",
            []
        )

    model_id, model_version_id, error = _resolve_civitai_input_ids(
        model_url_or_id,
        model_id,
        model_version_id
    )
    if error:
        util.printD(error)
        return ({}, model_id or "", model_version_id or "", error, [])

    (
        model_id,
        model_version_id,
        version_info,
        error,
        fallback_version
    ) = _get_exact_version_info(model_id, model_version_id)

    if error:
        util.printD(error)
        return ({}, model_id or "", model_version_id or "", error, [])

    preview_state = {
        "model_type": model_type,
        "model_name": model_name,
        "model_path": model_path,
        "model_id": model_id,
        "model_version_id": model_version_id,
    }

    preview = _format_match_preview(
        model_path,
        model_type,
        model_id,
        model_version_id,
        version_info,
        fallback_version
    )

    preview_images = _get_version_preview_images(version_info)

    return (
        preview_state,
        model_id,
        model_version_id,
        preview,
        preview_images
    )


def apply_model_info_by_input(
    preview_state,
    model_type,
    model_name,
    model_id,
    model_version_id
):
    """
    Write metadata only for the exact model/version pair that was previewed.
    """
    model_id = _normalize_civitai_id(model_id)
    model_version_id = _normalize_civitai_id(model_version_id)

    if not preview_state:
        yield "Preview the Civitai match before writing."
        return

    expected = {
        "model_type": model_type,
        "model_name": model_name,
        "model_id": model_id,
        "model_version_id": model_version_id,
    }
    for key, value in expected.items():
        if _normalize_civitai_id(preview_state.get(key)) != _normalize_civitai_id(value):
            yield "The model selection or ids changed after the preview. Preview the match again before writing."
            return

    model_path = model.get_model_path_by_type_and_name(model_type, model_name)
    if model_path is None:
        yield "Could not get local model path."
        return

    if os.path.realpath(model_path) != os.path.realpath(preview_state.get("model_path", "")):
        yield "The local model path changed after the preview. Preview the match again before writing."
        return

    (
        actual_model_id,
        actual_version_id,
        model_info,
        error,
        _
    ) = _get_exact_version_info(model_id, model_version_id)

    if error:
        util.printD(error)
        yield error
        return

    if actual_model_id != model_id or actual_version_id != model_version_id:
        yield "Civitai returned different ids than the previewed selection. Nothing was written."
        return

    parent = model_info.get("model", {}) or {}
    civitai_type_name = parent.get("type")
    civitai_local_type = civitai.MODEL_TYPES.get(civitai_type_name)

    if not _model_types_compatible(model_type, civitai_local_type):
        yield (
            f"Model type mismatch: local type is {model_type}, but Civitai "
            f"reports {civitai_type_name} ({civitai_local_type}). Nothing was written."
        )
        return

    max_size_preview = util.get_opts("ch_max_size_preview")
    nsfw_preview_threshold = util.get_opts("ch_nsfw_threshold")

    model.process_model_info(
        model_path,
        model_info,
        model_type,
        force_civitai=True
    )

    yield from civitai.get_preview_image_by_model_path(
        model_path,
        max_size_preview,
        nsfw_preview_threshold,
        images=model_info.get("images", [])
    )

    yield (
        f"Done. Wrote Civitai metadata for modelId {model_id}, "
        f"modelVersionId {model_version_id}."
    )

def build_article_from_version(version):
    """
    Builds the HTML for displaying new model versions to the user.

    return: html:str
    """
    (
        model_path, model_id, model_name, new_version_id,
        new_version_name, description, download_url,
        img_url, model_type
    ) = version

    thumbnail = ""
    if img_url:
        thumbnail = templates.thumbnail.substitute(
            img_url=img_url,
        )

    if download_url:
        # replace "\" to "/" in model_path for windows
        download_model_path = model_path.replace('\\', '\\\\')

        download_section = templates.download.substitute(
            new_version_id=new_version_id,
            new_version_name=new_version_name,
            model_path=download_model_path,
            model_type=model_type,
            download_url=download_url
        )

    else:
        download_section = templates.no_download.substitute(
            new_version_name=new_version_name,
        )

    description_section = ""
    if description:
        description_section = templates.description.substitute(
            description=util.safe_html(download_section),
        )

    article = templates.article.substitute(
        url=f'{civitai.URLS["modelPage"]}{model_id}',
        thumbnail=thumbnail,
        download=download_section,
        description=description_section,
        model_name=model_name,
        model_path=model_path
    )

    return article


def check_models_new_version_to_md(model_types:list) -> str:
    """
    check models' new version and output to UI as html doc
    return: html:str
    """
    new_versions = civitai.check_models_new_version_by_model_types(model_types, 0.2)

    if not new_versions:
        util.printD("Done: no new versions found.")
        return "No models have new versions"

    articles = []
    count = 0
    for index, new_version in enumerate(new_versions):
        article = build_article_from_version(new_version)
        articles.append(article)

    output = f"Found new versions for following models: <section>{''.join(articles)}</section>"

    count = index + 1

    if count != 1:
        util.printD(f"Done. Found {count} models that have new versions. Check UI for detail")
    else:
        util.printD(f"Done. Found {count} model that has a new version. Check UI for detail.")

    return output


def get_model_info_by_id(model_id:str) -> dict:
    """
    Retrieves model information necessary to populate HTML
    with Model Name, Model Type, valid saving directories,
    and available model versions.

    return: tuple or None
    """
    util.printD(f"Getting model info for: {model_id}")

    try:
        # download model info
        model_info = civitai.get_model_info_by_id(model_id)

        # parse model type, model name, subfolder, version from this model info
        # get model type
        civitai_model_type = model_info["type"]

        if civitai_model_type not in civitai.MODEL_TYPES:
            util.printD(f"This model type is not supported: {civitai_model_type}")
            return None

        model_type = civitai.MODEL_TYPES[civitai_model_type]

        # get model type
        model_name = model_info["name"]

        # get version lists
        model_versions = model_info["modelVersions"]

    except (KeyError, ValueError, TypeError) as e:
        util.printD(f"An error occurred while attempting to process model info: \n\t{e}")
        return None

    filenames = []
    files = []
    version_strs = []
    base_models = []
    previews = {}
    for version in model_versions:
        # version name can not be used as id
        # version id is not readable
        # so , we use name_id as version string
        version_str = f'{version["name"]}_{version["id"]}'

        filename = ""
        try:
            for filedata in version["files"]:
                if filedata["type"] == "Model":
                    filename = filedata["name"]

        except (ValueError, KeyError):
            pass

        files.append(version["files"])
        filenames.append(filename)
        version_strs.append(version_str)
        base_models.append(version.get("baseModel", None))
        previews[version_str]: list = version.get("images", [])

    # get folder by model type
    folder = model.folders[model_type]

    # get subfolders
    subfolders = ["/"] + util.get_subfolders(folder)

    # msg = util.indented_msg(f"""
    #     Got following info for downloading:
    #     {model_name=}
    #     {model_type=}
    #     {version_strs=}
    #     {base_models=}
    #     {subfolders=}
    #     {previews=}
    # """)
    # util.printD(msg)

    return {
        "model_info": model_info,
        "model_name": model_name,
        "model_type": model_type,
        "files": files,
        "filenames": filenames,
        "subfolders": subfolders,
        "version_strs": version_strs,
        "base_models": base_models,
        "previews": previews
    }


def get_ver_info_by_ver_str(version_str:str, model_info:dict) -> dict:
    """
    get version info by version string

    return: version_info:dict
    """

    if not (version_str and model_info):
        output = util.indented_msg(
            f"""
            Missing Parameter:
            {model_info=}
             {version_str=}
            """
            )
        util.printD(output)
        return None

    # get version list
    model_versions = model_info.get("modelVersions", None)
    if model_versions is None:
        util.printD("modelVersions is Empty")
        return None

    # find version by version_str
    version = None
    for ver in model_versions:
        # version name can not be used as id
        # version id is not readable
        # so , we use name_id as version string
        ver_str = f'{ver["name"]}_{ver["id"]}'
        if ver_str == version_str:
            # find version
            version = ver
            break

    if not (version and ("id" in version)):
        util.printD(f"can not find version or id by version string: {version_str}")
        return None

    return version


def get_id_and_dl_url_by_version_str(version_str:str, model_info:dict) -> tuple:
    """
    get download url from model info by version string
    return - (version_id, download_url)
    """
    if not (version_str and model_info):
        output = util.indented_msg(f"""
            Missing Parameter:
            {model_info=}
            {version_str=}
        """)
        util.printD(output)
        return (False, output)

    # get version list
    model_versions = model_info.get("modelVersions", None)
    if model_versions is None:
        util.printD("modelVersions is Empty")
        return (False, output)

    # find version by version_str
    version = None
    for ver in model_versions:
        # version name can not be used as id
        # version id is not readable
        # so , we use name_id as version string
        ver_str = f'{ver["name"]}_{ver["id"]}'
        if ver_str == version_str:
            # find version
            version = ver
            break

    version_id = None
    download_url = None
    if version:
        download_url = version.get("downloadUrl", None)
        version_id = version.get("id", None)

    if None in [version, version_id, download_url]:
        output = util.indented_msg(f"""
            Invalid Version Information:
            {version=}
            {version_id=}
            {download_url=}
        """)
        util.printD(output)
        return (False, output)

    util.printD(f"Get Download Url: {download_url}")

    return (version_id, download_url)

def parse_file_info(file_info, basename):
    """
        returns data required to download a file from civitai
    """

    download_url = file_info.get("downloadUrl", None)
    if download_url is None:
        return None

    filetype = file_info["type"]
    filename = file_info["name"]
    if basename and not filetype == "VAE":
        filename = f"{basename}.{filename.split('.')[-1]}"

    return {
        "url": download_url,
        "filename": filename,
        "type": filetype
    }


def download_files(filename, model_folder, ver_info, headers, filetypes, dl_all, duplicate):
    """
    get download urls from files info
    some model versions have multiple files
    """

    version_id = ver_info["id"]
    model_id = ver_info["model_id"]

    model_ids = {
        "model": model_id,
        "version": version_id
    }

    downloads = []

    # check if this model already exists
    result = civitai.search_local_model_info_by_version_id(model_folder, model_ids)
    if result:
        output = f"This model version already exists at `{result}`"
        util.printD(output)
        yield (False, output)

    for file_info in ver_info.get("files", {}):
        if not dl_all:
            if not file_info["type"] in filetypes:
                continue

        dl_info = parse_file_info(file_info, filename)

        if dl_info:
            downloads.append(dl_info)

    if len(downloads) == 0:
        dl_info = parse_file_info(ver_info, filename)
        if dl_info:
            downloads.append(dl_info)

    # download
    success = False
    output = ""
    filepath = None
    total = len(downloads)
    errors = []
    errors_count = 0
    snippet = None

    for index, dl_info in enumerate(downloads):

        url = dl_info["url"]
        if errors_count > 0:
            snippet = f"{errors_count}/{total} files failed"

        dl_folder = model_folder
        if dl_info["type"] == "VAE":
            dl_folder = model.folders["vae"]

        # webui visible progress bar
        for result in downloader.dl_file(
            url, filename=dl_info["filename"], folder=dl_folder, duplicate=duplicate,
            headers=headers
        ):
            if not isinstance(result, str):
                success, output = result
                break

            output = f"{result} | {index+1}/{total} files"
            if snippet:
                " | ".join([output, snippet])

            yield output

        if not success:
            errors.append(downloader.error(url, output))
            errors_count += 1
            continue

        if dl_info["type"] == "Model":
            filepath = output

    additional = None
    if errors_count > 0:
        additional = "\n\t".join(errors)

        if errors_count == total:
            yield (False, additional)
            return

    yield (True, filepath, additional)


def download_one(filename, model_folder, ver_info, headers, duplicate):
    """
    only download one file
    get download url
    """

    download_url = ver_info["downloadUrl"]

    output = ""
    if not download_url:
        output = "Failed to find a download url"
        util.printD(output)
        yield (False, output)

    # download
    success = False
    for result in downloader.dl_file(
        download_url, filename=filename, folder=model_folder,
        duplicate=duplicate, headers=headers
    ):
        if not isinstance(result, str):
            success, output = result
            break

        yield result

    if not success:
        downloader.error(download_url, output)
        yield (False, output)

    yield (True, output)


def dl_model_by_input(
    ch_state:dict,
    model_type:str,
    subfolder_str:str,
    version_str:str,
    filename:str,
    file_ext:str,
    dl_all:bool,
    duplicate:str,
    preview:str,
    *args
) -> str:
    """ download model from civitai by input
        output to markdown log
    """

    model_info = ch_state["model_info"]
    max_size_preview = util.get_opts("ch_max_size_preview")
    nsfw_preview_threshold = util.get_opts("ch_nsfw_threshold")

    if not (model_info and model_type and subfolder_str and version_str):
        output = util.indented_msg(f"""
            Missing Required Parameter in dl_model_by_input. Parameters given:
            {model_type=}*
            {subfolder_str=}*
            {version_str=}*
            {filename=}
            {file_ext=}
            {duplicate=}
        """)

        # Keep model info away from util.indented_msg
        # which can screw with complex strings
        output = f"{output}\n    {model_info=}*\n    * Required"
        util.printD(output)
        yield output
        return

    # get model root folder
    if model_type not in model.folders:
        output = f"Unsupported model type: {model_type}"
        util.printD(output)
        yield output
        return

    folder = ""
    subfolder = ""
    output = ""
    version_info = None

    filetypes = []
    for filetype, will_dl in zip(civitai.FILE_TYPES, args):
        if will_dl:
            filetypes.append(filetype)

    model_root_folder = model.folders[model_type]

    if not os.path.exists(model_root_folder):
        # Model directories may not exist by default
        os.mkdir(model_root_folder)

    # get subfolder
    if subfolder_str in ["/", "\\"]:
        subfolder = ""
    elif subfolder_str[:1] in ["/", "\\"]:
        subfolder = subfolder_str[1:]
    else:
        subfolder = subfolder_str

    # get model folder for downloading
    folder = os.path.join(model_root_folder, subfolder)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            output = f"Could not create directory: {subfolder}."
            util.printD(output)

            yield output
            return

    if not os.path.isdir(folder):
        subfolders = []
        for file in os.listdir(model_root_folder):
            if os.path.isdir(os.path.join(model_root_folder, subfolder)):
                subfolders.append(file)
        if len(subfolders) == 0:
            subfolders = ["No subfolders exist."]
        subfolders = "\n\t".join(subfolders)

        output = f"Model folder is not a dir: {subfolder}. Available subfolders: \n\t{subfolders}"
        util.printD(output)

        yield output
        return

    # get version info
    ver_info = get_ver_info_by_ver_str(version_str, model_info)
    ver_info["model_id"] = model_info["id"]
    if not ver_info:
        output = "Failed to get version info, check console log for detail"
        util.printD(output)
        yield output
        return

    headers = {
        "content-type": "application/json"
    }
    api_key = util.get_opts("ch_civiai_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    additional = None
    for result in download_files(filename, folder, ver_info, headers, filetypes, dl_all, duplicate):
        if not isinstance(result, str):
            if len(result) > 2:
                success, output, additional = result
            else:
                success, output = result

            break

        yield result

    if not success:
        yield output
        return

    # get version info
    version_info = civitai.get_version_info_by_version_id(ver_info["id"])
    model.process_model_info(output, version_info, model_type)

    # then, get preview image + webui-visible progress
    for result in civitai.get_preview_image_by_model_path(
        output,
        max_size_preview,
        nsfw_preview_threshold,
        preferred_preview=preview
    ):
        yield f"Downloading model preview:\n{result}"

    output = f"Done. Downloaded to: {output}"
    if additional:
        output = f"{output}. Additionally, the following failures occurred: \n{additional}"
    util.printD(output)
    yield output
