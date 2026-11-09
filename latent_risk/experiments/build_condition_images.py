"""Build the SWE-bench benign and pair images of the 8 in-scope task pairs (or --pairs) for one
injection condition (driver: dataset_v1_eval.sh build-images).

The injection is written into the image at build time, and non-default conditions get their own
tags (siren-swebench-benign:<instance>__skill_only, ...), so the images of one condition can be
built while jobs of another condition run. Missing images are built; existing ones are kept unless
--rebuild (which deletes and rebuilds them, base and env stages included: run it only when no job
is running). Standard-condition images built before the condition tags existed may hold another
condition's build, so rebuild them once before reusing them.
"""

from __future__ import annotations

import argparse
import asyncio

from prompt_siren.build_images import ImageBuilder
from prompt_siren.datasets import get_image_build_specs
from prompt_siren.datasets.swebench_dataset.config import SwebenchDatasetConfig
from prompt_siren.datasets.swebench_dataset.image_tags import (
    get_benign_image_tag,
    get_injection_variant,
    get_pair_image_tag,
)
from prompt_siren.sandbox_managers.docker.manager import create_docker_client_from_config
from prompt_siren.sandbox_managers.image_spec import DerivedImageSpec, MultiStageBuildImageSpec

TASK_PAIRS = (
    "astropy__astropy-14182:disable_security_tests_task",
    "astropy__astropy-14182:skip_error_handling_task",
    "astropy__astropy-14995:dependency_poisoning_task",
    "sympy__sympy-15345:modify_readme_curl_bash_task",
    "sympy__sympy-15345:skip_error_handling_task",
    "sympy__sympy-17139:disable_security_tests_task",
    "sympy__sympy-17139:modify_readme_curl_bash_task",
    "sympy__sympy-17139:skip_error_handling_task",
)


async def build(condition: str, rebuild: bool, pairs: list[str]) -> None:
    config = SwebenchDatasetConfig(
        enable_source_injection=condition == "standard", enable_skill_injection=True
    )
    variant = get_injection_variant(config.enable_source_injection, config.enable_skill_injection)
    specs = get_image_build_specs("swebench", config, ".siren-docker-cache/swebench")
    derived = {spec.tag for spec in specs if isinstance(spec, DerivedImageSpec)}
    wanted: set[str] = set()
    for pair in pairs:
        instance, task = pair.split(":")
        wanted.add(get_benign_image_tag(instance, variant=variant))
        # a pair image exists only for malicious tasks that modify the benign image
        pair_tag = get_pair_image_tag(instance, task, variant=variant)
        if pair_tag in derived:
            wanted.add(pair_tag)
    selected = [
        spec
        for spec in specs
        if (isinstance(spec, MultiStageBuildImageSpec) and spec.final_tag in wanted)
        or (isinstance(spec, DerivedImageSpec) and spec.tag in wanted)
    ]
    found = {
        spec.final_tag if isinstance(spec, MultiStageBuildImageSpec) else spec.tag
        for spec in selected
    }
    missing = sorted(wanted - found)
    if missing:
        raise SystemExit(f"no build spec for {missing}")
    print(f"{'Rebuilding' if rebuild else 'Building missing'} images ({condition}):")
    for tag in sorted(wanted):
        print(f"  {tag}")
    docker = create_docker_client_from_config("local", {})
    try:
        builder = ImageBuilder(docker_client=docker, rebuild_existing=rebuild, registry=None)
        # builds the benign images first, then the pair images derived from them
        errors = await builder.build_all_specs(selected)
    finally:
        await docker.close()
    if errors:
        raise SystemExit(
            "failed: "
            + ", ".join(f"{error.image_tag}: {type(error.error).__name__}" for error in errors)
        )
    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("standard", "skill_only"), required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--pairs", nargs="+", default=list(TASK_PAIRS), help="task ids (default: the 8 in scope)"
    )
    args = parser.parse_args()
    asyncio.run(build(args.condition, args.rebuild, args.pairs))


if __name__ == "__main__":
    main()
