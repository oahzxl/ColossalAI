# Copyright 2023 Together Computer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""RedPajama: An Open-Source, Clean-Room 1.2 Trillion Token Dataset."""


import json

import datasets
import traceback
import os

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset.
"""

_URL_LISTS = {
    # "arxiv": "new_urls/arxiv.txt",
    # "book": "new_urls/book.txt",
    "c4_1": "new_urls/c4_1.txt",
    "c4_2": "new_urls/c4_2.txt",
    "c4_3": "new_urls/c4_3.txt",
    # "common_crawl": "new_urls/common_crawl.txt",
    # "github": "new_urls/github.txt",
    "test": "new_urls/test.txt",
    # "stackexchange": "new_urls/stackexchange.txt",
    # "wikipedia": "new_urls/wikipedia.txt",
}
_URL_BASE = 'https://data.together.xyz/redpajama-data-1T/v1.0.0'

_DATA_DIR = "/data/personal/nus-zxl/VerticalMoE/data_prepare/downloads"

class RedPajama60BConfig(datasets.BuilderConfig):
    """BuilderConfig for RedPajama sample."""

    def __init__(self, *args, subsets, **kwargs):
        """BuilderConfig for RedPajama.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RedPajama60BConfig, self).__init__(**kwargs)
        self.subsets = subsets


class RedPajama60B(datasets.GeneratorBasedBuilder):
    """RedPajama: Reproducing the LLaMA training dataset of over 1.2 trillion tokens. Version 1.0.0."""

    BUILDER_CONFIGS = [
        RedPajama60BConfig(
            name = 'default',
            subsets = list(_URL_LISTS.keys()),
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T",
        ),

        # RedPajama60BConfig(
        #     name = 'arxiv',
        #     subsets = ['arxiv'],
        #     version=datasets.Version("1.0.0", ""),
        #     description="RedPajama1T arxiv subset",
        # ),
        
        RedPajama60BConfig(
            name = 'test',
            subsets = ['test'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T arxiv subset",
        ),

        # RedPajama60BConfig(
        #     name = 'book',
        #     subsets = ['book'],
        #     version=datasets.Version("1.0.0", ""),
        #     description="RedPajama1T book subset",
        # ),

        RedPajama60BConfig(
            name = 'c4_1',
            subsets = ['c4_1'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T c4 subset",
        ),
        RedPajama60BConfig(
            name = 'c4_2',
            subsets = ['c4_2'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T c4 subset",
        ),
        RedPajama60BConfig(
            name = 'c4_3',
            subsets = ['c4_3'],
            version=datasets.Version("1.0.0", ""),
            description="RedPajama1T c4 subset",
        ),

        # RedPajama60BConfig(
        #     name = 'common_crawl',
        #     subsets = ['common_crawl'],
        #     version=datasets.Version("1.0.0", ""),
        #     description="RedPajama1T common crawl subset",
        # ),

        # RedPajama60BConfig(
        #     name = 'github',
        #     subsets = ['github'],
        #     version=datasets.Version("1.0.0", ""),
        #     description="RedPajama1T github subset",
        # ),

        # RedPajama60BConfig(
        #     name = 'stackexchange',
        #     subsets = ['stackexchange'],
        #     version=datasets.Version("1.0.0", ""),
        #     description="RedPajama1T stackexchange subset",
        # ),

        # RedPajama60BConfig(
        #     name = 'wikipedia',
        #     subsets = ['wikipedia'],
        #     version=datasets.Version("1.0.0", ""),
        #     description="RedPajama1T wikipedia subset",
        # ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "meta": datasets.Value("string"),
                    "red_pajama_subset": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        url_lists = dl_manager.download_and_extract({
            subset: _URL_LISTS[subset] for subset in self.config.subsets
        })

        urls = {}

        for subset, url_list in url_lists.items():
            with open(url_list, encoding="utf-8") as f:
                urls[subset] = [line.strip() for line in f]

        if _DATA_DIR is not None:
            print(f'Reading data from {_DATA_DIR}')
            url_prefix_slashes = len(_URL_BASE.split('/'))
            downloaded_files = {
                subset: [
                    os.path.join(_DATA_DIR, *url.split('/')[url_prefix_slashes:])
                    for url in url_list
                ]
                for subset, url_list in urls.items()
            }
        else:
            downloaded_files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs = {
                    "files": {
                        subset: downloaded_files[subset]
                        for subset in self.config.subsets
                    }
                }
            )
        ]

    def _generate_examples(self, files):
        """This function returns the examples in the raw (text) form."""
        key = 0
        for subset in files:
            for path in files[subset]:
                with open(path, encoding="utf-8") as f:
                    for i, row in enumerate(f):
                        try:
                            data = json.loads(row)
                            if "meta" not in data:
                                text = data["text"]
                                del data["text"]
                                yield key, {
                                    "text": text,
                                    "meta": json.dumps(data),
                                    "red_pajama_subset": subset,
                                }
                            else:
                                yield key, {
                                    "text": data["text"],
                                    "meta": data["meta"],
                                    "red_pajama_subset": subset,
                                }
                            key += 1
                        except Exception as e:
                            print(f'Subset: {subset}')
                            print(f'Path: {path}')
                            print(f'Row: {row}')
                            traceback.print_exc()

                            raise e

