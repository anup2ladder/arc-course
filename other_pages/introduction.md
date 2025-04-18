# Introduction

Hi! I am Fausto Carcassi and this is the material for my short introduction to program synthesis, in the context of the Abstraction and Reasoning Corpus (ARC) challenge. I initially taught this as a 1-month course for the Master of Logic at the University of Amsterdam, where I work. We covered the content in this website in the first week, and then the students organized into groups and developed their own attempt at the ARC challenge. This was a lot of fun! Greg Kamradt from the ARC foundation heard about this little course and asked if I would be interested in turning it into an online course, which I decided to do; You are seeing the result. This course is meant for people who are curious about the ARC challenge and want to read the current literature or develop their own approach, but lack the formal background needed to do so. This is not an in-depth treatment of program synthesis or ARC: there is much I won't cover and I will not be very formal. Nonetheless, it will at least give you an idea of the options, and how the ARC challenge fits within the wider field of program synthesis.

The main design decision behind this course was whether to focus on the tools and techniques relevant to the 2024 edition of the ARC challenge, or on more foundational topics, seeing ARC as a specific application. In the end, I decided to go with the latter, for three reasons. First, the ARC challenge is constantly stimulating new ideas and approaches, and the tools and techniques that are relevant today may not be relevant tomorrow. Second, it is hard to understand the current approaches without knowing what they are based on. Finally and most importantly, the spirit of the ARC challenge is to demonstrate the importance of in-context rule learning from sparse data for intelligence, and by covering more basic topics I hope that you can familiarize yourself with the problem of _searching for a program_ which underlies the ARC challenge. I hope this decision will make the course more useful to you in the long run. Still, a lot of the material is framed with the ARC challenge in mind, and the last module is a review of some recent approaches.

This mini-course is organized into five modules. Module 1 is a general introduction to the problem of program synthesis and its relation to intelligence. Module 2 covers some formal tools that are commonly used to develop program synthesis systems. Module 3 and 4 summarize a variety of ways to approach the problem of searching for programs, respectively with symbolic and neuro(symbolic) methods. Module 5 is a review of some exciting approaches to the ARC challenge. 

All modules except for 1 consist of a video lecture and a programming lab. The programming lab is a Jupyter notebook that you can download and run locally on your computer (you can also run it on google Colab in principle but I haven't fixed the installs/imports yet). If there is interest, I will also record videos walking step-by-step through the labs (let me know with the feedback form below, in a youtube comment, or just via the email on my website). This course is work in progress and was done somewhat quickly at various levels of exhaustion and in between other obligations, so it is still rough around the edges. Nonetheless, I hope you find it useful. Please let me know if you have any comments, thoughts, questions, and/or feedback about the course. 

Finally, I wanted to mention some some impressive & awesome people at the intersection of cognition and program synthesis that you might want to keep on your radar (in random order, mostly young scholars): 

- [Steven Piantadosi](https://colala.berkeley.edu/people/piantadosi/)
- [Bonan Zhao](https://zhaobn.github.io/)
- [Tom Griffiths](https://cocosci.princeton.edu/tom/index.php)
- [Lio Wong](https://web.mit.edu/zyzzyva/www/academic.html)
- [Gabe Grand](https://www.gabegrand.com/)
- [Kevin Ellis](https://www.cs.cornell.edu/~ellisk/)
- [Josh Rule](https://joshrule.com/)
- [Tan Zhi Xuan](https://ztangent.github.io/)
- [Tomer Ullman](https://www.tomerullman.org/)
- [Neil Bramley](https://www.bramleylab.ppls.ed.ac.uk/member/neil/)

The form below is completely anonymous and no identifiable information will be collected or stored. I will use the feedback to improve the course and make it more useful to you. Thank you!

> **NOTE** I know, this is the internet and this form is anonymous, but please still be respectful.

<iframe id="qualtrics-iframe" src="https://uva.fra1.qualtrics.com/jfe/form/SV_eeyyaul3x2iKHLU" height="500px" width="600px"></iframe>

