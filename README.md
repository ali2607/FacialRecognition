# Reconnaissance d'Émotions via Expressions Faciales

Ce projet a pour objectif de détecter et classifier les émotions à partir des expressions faciales. Il combine la détection de visages avec la méthode Haar Cascade et un modèle de réseau de neurones convolutifs (CNN) entraîné sur un dataset (FER-2013 sur Kaggle.com) comprenant sept émotions : Angry, Disgust, Fear, Happy, Neutral, Sad et Surprise.

**Notes :** Ce projet était uniquement à but éducatif pour un projet scolaire, les résultats, la qualité du code et la précision des résultats peuvent être très largement améliorés.
## Prérequis
- Python 3
- Bibliothèques Python :
    - OpenCV
    - NumPy
    - TensorFlow / Keras
    - Keras Tuner
    - scikit-learn
    - matplotlib
    - seaborn
- Une webcam (pour le test en temps réel via live_cam_test.py)
- Une image de test placée dans le dossier FacesImages (par exemple test.jpg) pour le script picture_test.py

## Installation

1. **Cloner le repository**

2. **Installer les dépendances**

Il est recommandé de créer un environnement virtuel. Par exemple, avec venv.

## Utilisation
### Entrainement du Modèle
Pour entraîner le modèle et effectuer la recherche d'hyperparamètres via Keras Tuner, lancez :
```
python model_training.py
```

Le modèle entraîné sera sauvegardé sous le nom "emotion_detection.keras".

### Evaluation du Modèle 

Pour évaluer les performances du modèle sur le dataset de test et visualiser la matrice de confusion, exécutez :

```
python metrics.py
```


Ce script affiche le taux de perte, l'exactitude, ainsi qu'un rapport de classification détaillé.

### Reconnaissance en Temps Réel

Pour lancer la détection d'émotions en temps réel via votre webcam :
```
python live_cam_test.py
```
Appuyez sur la touche x pour quitter l'application.

### Test sur image
Pour tester la reconnaissance sur une image statique (par exemple test.jpg placé dans le dossier FacesImages), lancez :
```
python picture_test.py
```
