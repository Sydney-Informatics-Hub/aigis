<p align="center">
    <h1 align="center">AIGIS</h1>
</p>
<p align="center">
    <em>AI annotation, segmentation, and conversion tools for GIS imagery</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Sydney-Informatics-Hub/aigis?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Sydney-Informatics-Hub/aigis?style=flat&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Sydney-Informatics-Hub/aigis?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Sydney-Informatics-Hub/aigis?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	</p>
<hr>

![aigis](docs/content/aigis.png)

# aigis

`aigis` is a comprehensive toolkit for aerial and satellite imagery acquisition, processing, annotation, and analysis using artificial intelligence. This repository contains three main components:

1. **annotate:** Tools for annotating aerial imagery data.
2. **convert:** Utilities for converting aerial imagery data to various formats, including COCO, GeoJSON, etc.
3. **segment:** Scripts and notebooks for segmenting aerial imagery using deep learning models.

## Installation

Clone the repository:

```bash
git clone https://github.com/Sydney-Informatics-Hub/aigis.git
cd aigis
```

## Usage

### annotate
Scripts for annotating aerial imagery data. Detailed usage instructions can be found in the aerial_annotation directory.

### convert
Tools for converting aerial imagery data to various formats. For detailed instructions, refer to the aerial_conversion directory.

### segment
Scripts for segmenting aerial imagery using deep learning models. Refer to the aerial_segmentation directory for more details.

![Sydney city](docs/content/sydney_city_geospatial.jpeg)

### Recommended System Requirements

All tools were built for Linux or MacOS systems.

High resolution aerial imagery quickly becomes very computationally intensive to work with, so we recommend using high powered workstations or cloud computing environments.

Segmentation model fine tuning and prediction are best with CUDA GPUs. An RTX 4090 or A100 is recommended for best performance. 
Models can be fine tuned on Google Colab free T4 GPUs, but larger datasets and longer runs should run on other compute platforms.

## Example Usage

### Building Segmentation

We've used the full `aigis` toolkit to fine tune segmentation models to detect and segment building outlines in aerial imagery. Our fine tuned model was then run on 37,000 high reslution aerial images across the entire Greater Sydney Urban area in New South Wales, Australia. 

Our models's predictions are available as a shapefile in GeoJSON, with almost one million buildings (980k).

![Greater Sydney Buildings](docs/content/gsu_buildings.png)

### Tree Segmentation

WIP

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github/Sydney-Informatics-Hub/aigis/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github/Sydney-Informatics-Hub/aigis/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github/Sydney-Informatics-Hub/aigis/issues)**: Submit bugs found or log feature requests for Aigis.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/Sydney-Informatics-Hub/aigis
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

## About Us

This software was developed by the Sydney Informatics Hub, a core research facility of the University of Sydney.

The project team includes:

- Henry Lydecker
- Sahand Vahidnia
- Xinwei Luo
- Thomas Mauch

##  License

This project is protected under the [MIT Licence](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.


##  Acknowledgments

Acknowledgements are an important way for us to demonstrate the value we bring to your research. Your research outcomes are vital for ongoing funding of the Sydney Informatics Hub. If you make use of this software for your research project, please include the following acknowledgement:

>This research was supported by the Sydney Informatics Hub, a Core Research Facility of the University of Sydney.
