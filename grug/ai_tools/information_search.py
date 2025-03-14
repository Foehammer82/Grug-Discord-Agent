# TODO: copy over code for general searches, thinking of re-writing it to be a second graph that knows which TTRPG
#       the play is using and use tools (like AoN for pathfinder) that are specific to that TTRPG.
# TODO: this is also where the RAG stuff comes in where admins can upload source material and have it be searchable.
# TODO: create a slash-command that the user can use to set which TTTRPG they are playing and use that here when
#       looking up information.  it would also be good to send a warning or something if no source material is found
#       and give instructions for how to upload it.  should also have slash commands to perform crud operations on the
#       source material for their server/guild.
from elasticsearch import Elasticsearch
from langchain_core.tools import tool
from loguru import logger


@tool(parse_docstring=True)
def search_archives_of_nethys(search_string: str) -> list[dict]:
    """
    Searches the Elasticsearch index for entries matching the given search string within the
    [AON](https://2e.aonprd.com/) (Archives of Nethys) dataset.

    Args:
        search_string (str): The string to search for within the AON dataset.

    Returns:
        list[dict]: A list of dictionaries, each representing a cleaned-up search result. Each dictionary contains
        the keys:
            - name (str): The name of the entry.
            - type (str): The type of the entry (e.g., Ancestry, Class).
            - summary (str, optional): A summary of the entry, if available.
            - sources (list): The sources from which the entry is derived.
            - url (str): The URL to the detailed entry on the AON website.

    Note:
        This function requires the Elasticsearch Python client and assumes access to an Elasticsearch instance with
        the AON dataset indexed under the index named "aon".
    """
    logger.info(f"Searching AoN for: {search_string}")

    es = Elasticsearch("https://elasticsearch.aonprd.com/")

    es_response = es.search(
        index="aon",
        query={
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match_phrase_prefix": {"name.sayt": {"query": search_string}}},
                            {"match_phrase_prefix": {"text.sayt": {"query": search_string, "boost": 0.1}}},
                            {"term": {"name": search_string}},
                            {
                                "bool": {
                                    "must": [
                                        {
                                            "multi_match": {
                                                "query": word,
                                                "type": "best_fields",
                                                "fields": [
                                                    "name",
                                                    "text^0.1",
                                                    "trait_raw",
                                                    "type",
                                                ],
                                                "fuzziness": "auto",
                                            }
                                        }
                                        for word in search_string.split(" ")
                                    ]
                                }
                            },
                        ],
                        "must_not": [{"term": {"exclude_from_search": True}}],
                        "minimum_should_match": 1,
                    }
                },
                "boost_mode": "multiply",
                "functions": [
                    {"filter": {"terms": {"type": ["Ancestry", "Class"]}}, "weight": 1.1},
                    {"filter": {"terms": {"type": ["Trait"]}}, "weight": 1.05},
                ],
            }
        },
        sort=["_score", "_doc"],
        aggs={
            "group1": {
                "composite": {
                    "sources": [{"field1": {"terms": {"field": "type", "missing_bucket": True}}}],
                    "size": 10000,
                }
            }
        },
        source={"excludes": ["text"]},
    )

    results_raw = [hit["_source"] for hit in es_response.body["hits"]["hits"]]

    results_clean = [
        {
            "name": hit["name"],
            "type": hit["type"],
            "summary": hit["summary"] if "summary" in hit else None,
            # "overview_markdown": hit["markdown"] if "markdown" in hit else None,
            # "rarity": hit["rarity"] if "rarity" in hit else None,
            "sources": hit["source_raw"],
            "url": f"https://2e.aonprd.com{hit['url']}",
        }
        for hit in results_raw
    ]

    logger.info(
        f'Found {len(results_clean)} results from AoN for "{search_string}": '
        f'{[result["name"] for result in results_clean]}'
    )

    return results_clean
