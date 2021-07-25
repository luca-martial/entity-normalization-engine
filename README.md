[![GitHub Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]
[![GitHub Stars][stars-shield]][stars-url]
[![Contributors][contributors-shield]][contributors-url]


# Building an Entity Normalization Engine

This repository contains experiments in an attempt to build an entity normalization engine. The input to this engine is short strings that could encompass the following entities: company names, company addresses, serial numbers, physical goods and locations. 

⚠️ **This project is in progress, and by no means complete. Project progress and approach are presented at the end of this readme.**

## Repository Structure

Here is a list of the files contained in this repository:

- **[experiments.ipynb](https://github.com/luca-martial/entity-normalization-engine/tree/main/experiments.ipynb)**: Jupyter notebook with thought process laid out and experiments that lead to the final approach considered to create the engine
- **[data](https://github.com/luca-martial/entity-normalization-engine/tree/main/data)**: Folder with datasets used for the project
- **[utils](https://github.com/luca-martial/entity-normalization-engine/tree/main/utils)**: Folder with collection of functions that will be created from experiments
- **[requirements.txt](https://github.com/luca-martial/entity-normalization-engine/blob/main/requirements.txt)**: Required libraries to install for the project

## Installation & Usage

Create a new environment and activate it:

```
conda create --name vectorai python=3.9.6 && conda activate vectorai
```

Install the required libraries:

```
pip install -r requirements.txt
```

## Contributing

Here's how to add a contribution:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingContribution`)
3. Commit your Changes (`git commit -m 'Add some AmazingContribution'`)
4. Push to the Branch (`git push origin feature/AmazingContribution`)
5. Open a Pull Request

## Reporting Issues

Does something seem off? Make sure to [report it](https://github.com/luca-martial/entity-normalization-engine/issues).

## Project Approach & Progress

### Task Description:

**Part 1**: Build a system that can identify unique entities for each category above. Some of these will be trivial (remove spaces, edit distance) while others are more complicated and will need a trained model / some other form of knowledge and guidance.

**Part 2**: Build a system that receives strings one by one and groups together any entities that have passed through the system previously. Check the latest sample received, scan the entries already received, identify if the entity is a duplicate and then add it to a cluster / create a new cluster depending on the result.

Here are fictional examples of the strings we will be dealing with:

Company names: “Marks and Spencers Ltd”, “M&S Limited”, “NVIDIA Ireland”
Company addresses: “SLOUGH SE12 2XY”, “33 TIMBER YARD, LONDON, L1 8XY”, “44 CHINA ROAD, KOWLOON, HONG KONG”
Serial numbers: “XYZ 13423 / ILD”, “ABC/ICL/20891NC”
Physical Goods: “HARDWOOD TABLE”, “PLASTIC BOTTLE”
Locations: “LONDON”, “HONG KONG”, “ASIA”

### Approach & Progress

✔️ **Part 1**: Category-Specific Normalization Engines

The general, theoretical approach here is:

Step 1 - Text preprocessing: standardizing lettercase, punctuation, whitespace, accented/special characters, legal control terms (Ltd, Co, etc) depending on which category we are dealing with

Step 2 - Entity clustering: use string_grouper library that uses TF-IDF (Term Frequency multiplied by Inverse Document Frequency) with N-Grams to calculate cosine similarities within a single Series of strings and groups them by assigning to each string one string from the group chosen as the group-representative for each group of similar strings found.

Note: Original idea was to create for each category string similarity matrix with similarity metric such as levenshtein distance, Jaro-Winkler or caverphone algorithm. Next, use clustering algorithm to cluster entities; affinity propagation seems to be standard for this task. For each unique entity (cluster) assign substring with the longest string length as the standard name for that cluster. I experimented with these approach but found the chosen approach to be quicker to execute, more legible and have better performance.

*Remaining task: generate python scripts for each category, make sure code is tested and ready to be used.*

**Part 2**: General Normalization Engine

My first thought in this part was to use the approach outlined in part 1, with the addition of Named Entity Recognition (NER) right after the text pre-processing step. This would allow to separate all strings into their respective categories (serial numbers, physical goods, locations, company addresses, company names) and would allow us to use the category-specific normalization engines that were built in part 1. The issues are that NER works best when words have a context within a sentence and entity types such as serial numbers and addresses would have to be custom-trained. We would also still be relying on the rule-based cleaning engines developed in part 1 instead of making use of intelligent systems.

In thinking about a solution, I think it could be interesting to explore NER with the spaCy library. An open-source annotation tool can be used to create a dataset that we could use to fine-tune a spaCy NER model. When I say fine-tuning, there are 2 possibilities that both seem like risky hacks:

Option 1 - Creating new, custom entity types for the entities that don't yet have an entity type (serial number, address, physical goods) and updating a model with these new types. The obvious risk is that we will be run into issues of conflicting entity types. For example ADDRESS vs LOCATION.

Option 2 - This one sounds quite ridiculous. We do not create custom entity types and instead train the model to recognize serial numbers as persons (PER) for example, making sure each category (serial numbers, physical goods, locations, company addresses, company names) have a unique entity type to categorize them.

We could also decide to not fine tune and instead start with a blank model and teach it new entity types. Beyond this NER task, we need to start thinking about alternatives to the rule-based cleaning engines.

Other ideas to explore:

- Try brute-force approach of grouping all entities, completely ignoring semantics. This will give us a baseline of the performance to beat
- Explore use of Flair library for NER, confidence score is a feature that could be useful.
- [Wikipedia paragraph classification](https://github.com/yashsmehta/named-entity-normalization): Scrape a paragraph from wikipedia describing the entity in question, pass the paragraph through a language model (ie: BERT), feed embedding vector to a shallow classifier to make the prediction of what entity class the new entity belongs to. Could use fastai library for super quick setup/execution.
- Re-explore clustering-based methods for entity normalization.
- Explore viability of Named Entity Linking using Facebook's BLINK library or spaCy.
- Create large dataset for all categories, assemble from all datasets found online so far.
- Dealing with incoming strings: seems like the current approach will hold up with incoming strings, shouldn't worry too much about this.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/luca-martial/entity-normalization-engine.svg
[issues-url]: https://github.com/luca-martial/entity-normalization-engine/issues

[forks-shield]: https://img.shields.io/github/forks/luca-martial/entity-normalization-engine.svg
[forks-url]: https://github.com/luca-martial/entity-normalization-engine/forks

[stars-shield]: https://img.shields.io/github/stars/luca-martial/entity-normalization-engine.svg
[stars-url]: https://github.com/luca-martial/entity-normalization-engine/stargazers

[contributors-shield]: https://img.shields.io/github/contributors/luca-martial/entity-normalization-engine.svg
[contributors-url]: https://github.com/luca-martial/entity-normalization-engine/contributors
