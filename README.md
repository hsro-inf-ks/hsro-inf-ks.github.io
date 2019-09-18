# Wahlmodul: Kognitive Systems (KS)

_Vorlesung zum fachwissenschaftlichen Modul **Kognitive Systeme** im [Masterstudiengang Informatik](https://www.th-rosenheim.de/technik/informatik-mathematik/) an der [Hochschule Rosenheim](https://www.th-rosenheim.de)._				

## Empfohlene Literatur

- Stuart Russell, Peter Norvig: [Artificial Intelligence: A Modern Approach](https://www.amazon.de/Artificial-Intelligence-Modern-Approach-Global/dp/1292153962/), Global Edition (Englisch)

- Richard S. Sutton , Andrew G. Barto:  [Reinforcement Learning: An Introduction](https://www.amazon.de/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262039249/) (Adaptive Computation and Machine Learning) (Englisch)

- Sudharsan Ravichandiran: [Hands-On Reinforcement Learning with Python: Master reinforcement and deep reinforcement learning using OpenAI Gym and TensorFlow](https://www.amazon.de/Hands-Reinforcement-Learning-Python-reinforcement/dp/1788836529/)

- Maxim Lapan: [Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks, value iteration, policy gradients, TRPO, AlphaGo Zero and more (English Edition)](https://www.amazon.de/Deep-Reinforcement-Learning-Hands-Q-networks/dp/1788834240/)


## Inhalt
- **Einführung kognitive Systeme** ([Skript](/assets/00-einfuehrung/00-Einfuehrung.pdf), [Übung](https://github.com/hsro-inf-ks/00_uebung))

	Wir starten mit der Klärung von ein paar organisatorischen Dingen. Danach geht es darum eine Vorstellung zu bekommen, was "Kognitive Systeme" sind und worum es in diesem Modul überhaupt geht. Mit ein paar Szenarien motiviert, tauchen wir dann etwas tiefer in das Thema künstliche Intelligenz (KI; Artifical Intelligence AI) ein. Was kann es wohl heißen, daß ein System sehen, hören und lernen kann ... ganz im Sinne des Wortes "cognoscere" - erkennen, erlernen, erfahren.


- **Grundlagen lernender Systems** ([Skript](/assets/01-vorlesung/01-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/01_uebung))

	Nachdem das Gebiet in der ersten Vorlesung umrissen wurde, tauchen wir in dieser Vorlesung etwas tiefer in das Thema maschinelles Lernen ein. Dabei schauen wir uns an, was Klassifizierung bedeutet, was supervised, unsupervised und reinforcement Learning bedeutet.

	- Für die Lineare Regression wurde folgendes Jupyter Notebook verwendet: [Lineare_Regression](/assets/01-vorlesung/Linear_Regression.ipynb)
	- Für K-Means dieses Notebook: [k-means.ipynb](/assets/01-vorlesung/k-means.ipynb)


	Idealerweise programmieren sie einfach mal ein paar Ansätze selber: "Code your own Classifier!"


- **ANNs - Perceptron, Feed-Forward-Neural Networks und Backward Propagation** ([Skript](/assets/02-vorlesung/02-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/02_uebung))

	Wir picken uns einen speziellen Ansatz zum maschinellen Lernen heraus: Künstliche, neuronale Netze (ANNs). Hier werden wir verstehen, wie ein neuronales Netz funktioniert, welche Grenzen es hat und wie ein solches Netz tatsächlich lernt.

	In der Übung versuchen sie das einfach mal selber zu implementieren.


- **Convolutional Neural Networks (CNNs)** ([Skript](/assets/03-vorlesung/03-Vorlesung.pdf), [Fashion-MNIST-CNN-Keras](/assets/03-vorlesung/Fashion-CNN-Keras.ipynb), [Fashion-MNIST-MLNN-Keras](/assets/03-vorlesung/Fashion-MLNN-Keras.ipynb), [Übung](https://github.com/hsro-inf-ks/03_uebung))

	Ein ANN ist zwar ganz nett und die Grundlage für weitere Deep Neuronal Network (DNN) Ansätze. So hat sich ein Convolutional Neural Netwok (CNN) durchgesetzt, um Bilder zu klassifizieren oder evtl auch Sprache zu erkennen. Das ist sicherlich Grund genug, dass wir uns dem Thema CNN in einer Vorlesung mal ausführlich widmen.

	Als Übung werden wir uns mal mit Tensorflow und CNNs beschäftigen.


- **Recurrent Neural Networks (RNNs)** ([Skript](/assets/04-vorlesung/04-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/04_uebung))

	Ein weitere Vertiefung in ein spezielles DNN erwartet uns heute. Wir schauen in dieser Vorlesung auf Recurrent NEural Networks (RNNs). Diese speziellen Netze werden eingestzt, wann immer es um sequentielle Daten geht, z.B. Sprachverständnis, Sentiment-Analyse etc.

	- Wir hatten ein sehr einfaches vanilla RNN: [Vanilla RNN](/assets/04-vorlesung/RNN.ipynb)
	- Ein RNN, dass basierend auf TExten von Rochester University (ROCStories) Saetze generieren sollte: [ROCStories-RNN](/assets/04-vorlesung/ROCStories-RNN-Keras.ipynb)
	- Ein RNN, dass auf Charakter-Level Text generieren kann. Als Input hat hier ein Sonnet von Shakespeare geholfen: [RNN-Text](/assets/04-vorlesung/RNN-Text.ipynb)

	In der Übung schauen wir mal, was wir selber mit einem RNN anstellen können.


- **Transfer Learning** ([Skript], [Übung])

	Im Prinzip stahet das Thema "Transfer Learning" hier ein bisschen als Platzhalter für alle speziellen Themen im Bereich DNNs. Also schauen wir uns an, was Transfer Learning ist und wann es Sinn macht. Wenn Zeit ist, schauen wir evtl auch auf GANs oder Style Trasnfer Ansätze.

	Als Übung bauen wir unseren einfachen Image-Klassifizierer mit unseren eigenen Bilddaten (z.B. [CustomVision](http://customvision.ai)).


- **Einführung in Reinforcement Learning - RL Part I** ([Skript](/assets/06-vorlesung/06-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/06_uebung))

	Nach dem kurzen Break, starten wir in den nächsten lock, der sich dem Thema "Reinforcement Learnong" widmet. Zunächst ein bisschen Motivation und Überblick. Danach kommen wir vermutlich an Markov Decision Processes und Dynamic Programming (DP) nicht herum.

	Das ist viel Stoff für eine Vorlesung, deswegen wird die Übung auch wohl kurz ausfallen. Setup von OpenAI Gym.

	Jupyter Notebooks
	- [n-armed Bandit Problem](/assets/06-vorlesung/n-armed Bandit Problem.ipynb)
	- [Tic Tac Toe Bot](/assets/06-vorlesung/tic_tac_toe.ipynb)
	

- **RL Part II - Monte Carlo** ([Skript](/assets/07-vorlesung/07-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/07_uebung))

	Der nächste Schritt im RL Themengebiet führt uns zu Monte Carlo Methoden.

	Hier werden wir einfach ein paar coole Monte Carlo Ansätze selber programmieren. Warum? - Weil wir es können!


- **RL Part III - Temporal Difference Learning** ([Skript](/assets/08-vorlesung/08-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/08_uebung))

	Ein weiterer Block im RL Gebiet Temporal Difference Learning.


- **RL Part IV - Deep Reinforcement Learning** ([Skript](/assets/09-vorlesung/09-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/09_uebung))

	AlphaGo ist möglicherweise vielen bekannt. Die Leute von DeepMind haben dazu Deep Reinforcement Learning verwendet. Also sollten wir mal reinschauen, wie das so funktioniert.

	Damit schliessen wir dann den Block RL ab!


- **Signalvearbeitung** ([Skript](/assets/10-vorlesung/10-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/10_uebung))

	Nun wird es endlich Zeit sich wirklich dem Thema "Kognition" zu zuwenden. Dazu müssen wir uns leider ein paar Grundlagen ansehen. Nämlich: Wie werden Daten über Sensor an ein System weitergegeben und verarbeitet. Mathematisch reden wir hier von Faltung (incl. Dirac Funktion), Korrelation, Fouriertransformation, FFT,  und Spektrogrammen.

	In der Übung machen mir wuns die Hände schmutzig, in dem wir selber die eine oder andere Funktion implementieren, z.B.	Fourieranalyse mit Python oder Spektraldarstellung.


- **Speech Recognition** ([Skript](/assets/11-vorlesung/11-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/11_uebung))

	In dieser Vorlesung wird es um Spracherkennung gehen und wie sie grundsätzlich funtiioniert. Vermutlich wird hier das Thema "Transfer Learning" erneut relevant.

	In der Übung bauen wir unseren eigenen Spracherkenner (z.B. [Custom Speech](http://cris.ai)).


- **Intelligente Systeme in der Praxis** ([Skript](/assets/12-vorlesung/12-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/12_uebung))

	In dieser Vorlesung werden wir uns sehr praktisch dem Thema widmen, wie wir die Elemente, die wir bisher gelernt haben, in ein gesamt System giessen.

	In der Übung werden wir versuchen ein eigenes Audio-Klassifiezierungsmodell bauen.


- **Klausurvorbereitung und optimize your Inference Model** ([Skript](/assets/13-vorlesung/13-Vorlesung.pdf), [Übung](https://github.com/hsro-inf-ks/13_uebung))

	Was ist der Unterschied zwischen  Modell trainieren und Model ausführen. Wo kann ein Model ausgeführt werden und wie wird es für die darunter liegende Hardware optimiert.
	Wir schauen uns mal Paramter Pruning und Quantization an.

	In der Übung werden wir mit [OpenVino](https://01.org/openvinotoolkit) ein wenigen rumspielen und schauen, was es leistet.

- **Klausur** ([Probeklausur](/assets/probeklausur/probeklausur.pdf), [Musterloesung](/assets/probeklausur/musterloesung.pdf))

	In dieser Vorlesung schreiben wir die Klausur.



## Interessante Links:

- [KIT - Kognitive Systeme](https://www.youtube.com/watch?v=ryqsTcbO9nc&list=PLfk0Dfh13pBPWmCbvUYqNoP0qUVHdL9lC)
- [KIT - Demos](https://lecture-demo.ira.uka.de/)
- [David Silver: Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)
