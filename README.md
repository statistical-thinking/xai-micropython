![title](images/xai_micropython.png)

# XAI-MICROPYTHON
Explainable Artificial Intelligence (XAI) with MicroPython - a comprehensice codebook that covers relevant topics in AI **from statistical basics to machine learning and deep learning** and makes them **executable in MicroPython** (suitable for microcontrollers and devices like the PicoCalc) and live in your Browswer.

# Author
Prof. Dr. habil. Dennis Klinkhammer

# Requirements
MicroPython capable microcontrollers (or devices) or browser (e.g. Firefox).

# Educational Motivation
XAI-MICROPYTHON is an **educational framework** for developing and exploring explainable machine learning and deep learning algorithms on MicroPython-compatible microcontrollers. The framework is designed to **support teaching and learning of the fundamentals of artificial intelligence** on resource-constrained hardware. It introduces **regression, classification, clustering, and factor analysis** from machine learning, as well as the **architecture and operation of artificial neural networks** from deep learning. By running directly on microcontrollers, learners can engage with the underlying mathematical and statistical concepts **without depending on high-level machine learning frameworks**.

# Learning Objectives
XAI-MICROPYTHON aims to make the **mathematical and computational foundations of artificial intelligence accessible and transparent**. After working with the learning materials, learners should be able to:

* understand **fundamental statistical concepts** that form the basis of machine learning;
* explain the **basic principles of supervised and unsupervised machine learning**, including regression, classification, clustering, and factor analysis;
* understand the **architecture and basic operation of artificial neural networks**;
* implement and modify selected statistical, machine learning, and deep learning **algorithms in MicroPython**;
* **examine the underlying calculations and algorithmic steps** instead of treating artificial intelligence as a black box;
* **transfer these concepts** between browser-based learning environments and resource-constrained microcontroller hardware.

The individual examples are designed to progress from statistical foundations to machine learning and deep learning while keeping the implementation sufficiently transparent for learners to **inspect, modify, and experiment with the underlying code**.

# Live-Demo in Browser
XAI-MICROPYTHON can be used directly in the browser as **Web Version** or **Jupyer Notebook**! The easiest way is the web version, which explains all the code elements from statistics to machine learning and deep learning without any prerequisites: [WEB VERSION](https://statistical-thinking.de/xai/index.html). And the following link opens **JupyterLite** and the **Jupyter Notebook** with the name **xai-micropython.ipynb** can be called up by double-clicking. There is a concise explanation for each cell with MicroPython code, which can be extended as required in your own teaching settings. To execute the code, simply mark the corresponding cell and click on **Run Selected Cell**. Alternatively, all cells can be executed by clicking on **Run All Cells**. This is the link to the live-demo: [JUPYTER NOTEBOOK](https://statistical-thinking.github.io/xai-micropython)

# Execution on a Microcontroller
XAI-MICROPYTHON is so lightweight that it can even run on a microcontroller such as the **Raspberry Pi Pico 2**. If this is your first project with this kind of hardware, I recommend checking out the official [PICO DOCUMENTATION](https://pip-assets.raspberrypi.com/categories/610-raspberry-pi-pico/documents/RP-008276-DS-1-getting-started-with-pico.pdf). And if this is your first time with MicroPython (or Python as general purpose programming language), there is an official [PYTHON DOCUMENTATION](https://pip-assets.raspberrypi.com/categories/610-raspberry-pi-pico/documents/RP-008355-DS-1-raspberry-pi-pico-python-sdk.pdf) as well. In order to get started, simply transfer **xai_micropython.py** to the microcontroller via [THONNY](https://thonny.org/), for example, and execute it. In addition, devices such as the [PICOCALC](https://www.clockworkpi.com/picocalc) can be used to **create your own learning devices** with XAI-MICROPYTHON. 

# Community Guidelines
XAI-MICROPYTHON is designed to be used in (academic) teaching without any technical requirements. A preselection of statistical functions as well as machine learning and deep learning methods are available for this purpose. However, their functionality and the addition of further algorithms and methods are a primary goal of XAI-MICROPYTHON. Interested users can therefore contact me at any time in order to...

* make a corresponding contribution to the project,
* report issues or problems with the software,
* seek support.

# License
The current version 1.5.0 of XAI-MICROPYTHON is available under **MIT License**.
