# Documentation simple — Application de détection EPI (Streamlit)

## C'est quoi ?

Une petite application web (Streamlit) qui permet de regarder une caméra (ou une vidéo)
en direct et de voir, en temps réel, les équipements de protection individuelle (EPI)
détectés sur les personnes : casque, gilet, gants, lunettes, bottes — et aussi quand
ces équipements sont **manquants**.

Le modèle de détection (YOLOv8, exporté en `best.onnx`) a été entraîné dans le dossier
`training/` du projet ; cette application sert à le tester et à le démontrer en direct.

## Structure du dossier

```
yolo_streamlit_app/
├── app.py              # L'application principale (l'interface qu'on voit)
├── models/best.onnx    # Le modèle entraîné, utilisé en mode "Local"
├── scripts/export_onnx.py  # Convertit best.pt (training) en best.onnx
├── utils/
│   ├── inference.py    # Fait tourner le modèle (en local ou à distance)
│   ├── tracking.py     # Suit les objets d'une image à l'autre (anti faux positifs)
│   ├── streaming.py    # Ouvre la source vidéo (webcam / RTSP / fichier)
│   ├── benchmark.py    # Mesure la vitesse selon la résolution
│   └── zones.py        # Découpe une personne en 3 zones (tête / torse / jambes)
└── server/              # Petit serveur séparé (FastAPI) pour faire tourner le modèle à distance
```

## Comment la lancer

```bash
cd yolo_streamlit_app
pip install -r requirements.txt
python scripts/export_onnx.py   # génère models/best.onnx si besoin
streamlit run app.py
```

Une page web s'ouvre avec 3 onglets : **Live Detection**, **Benchmark**, **About / Deployment**.

## Les deux modes : Local vs Distant

Dans la barre latérale, on choisit comment le modèle tourne :

- **Local (this machine)** : le modèle tourne directement sur l'ordinateur qui fait
  tourner l'application. Simple, mais ça utilise le CPU/GPU de cette machine.
- **Remote (hosted demo)** : l'application envoie juste l'image à un serveur séparé
  (`server/app.py`, à héberger ailleurs, par exemple gratuitement sur Hugging Face Spaces)
  qui fait le calcul et renvoie le résultat. L'application elle-même reste très légère.

On change de mode simplement avec un bouton, sans toucher au code.

## Onglet "Live Detection"

C'est l'écran principal :

1. On choisit une **source vidéo** : caméra IP (RTSP), caméra IP (HTTP/MJPEG), webcam,
   ou un fichier vidéo à envoyer.
2. On clique sur **Start / Stop stream**.
3. L'image s'affiche avec des rectangles autour des personnes et des équipements
   détectés, avec leur nom et leur niveau de confiance.
4. À droite, un panneau montre des statistiques en direct : latence, FPS estimé,
   nombre de détections, usage CPU/RAM, etc.

### Réglages disponibles (barre latérale)

- **Confidence threshold** : à partir de quel niveau de certitude on affiche une
  détection. Plus on monte, moins il y a de fausses alertes, mais on peut rater
  de vraies détections peu sûres.
- **IoU / NMS threshold** : évite d'avoir deux rectangles sur le même objet.
- **Résolution** : plus petite = plus rapide mais un peu moins précis sur les petits
  objets. L'onglet Benchmark aide à choisir.
- **Target FPS** : limite le nombre d'images traitées par seconde.
- **ByteTrack** : donne un identifiant stable à chaque personne/objet suivi d'une
  image à l'autre, même s'il est caché une fraction de seconde.
- **Filtre temporel** (« min consecutive frames ») : une détection n'est affichée
  que si elle apparaît plusieurs images de suite. Ça élimine les faux positifs qui
  clignotent une seule image (reflet, ombre...).

### Zones EPI (tête / torse / jambes)

Chaque personne détectée est divisée en 3 bandes horizontales (haut/milieu/bas),
proportionnelles à sa propre taille dans l'image. Chaque équipement détecté est
ensuite associé à la zone où il se trouve. Si un équipement apparaît dans une zone
illogique (par exemple un « casque » détecté au niveau des pieds), l'application
l'affiche en **rouge avec un (!)** — c'est un signal qu'il s'agit probablement d'une
fausse détection plutôt qu'un vrai réglage de sécurité à corriger.

## Onglet "Benchmark"

Prend quelques images de la source actuelle et fait tourner le modèle à plusieurs
résolutions (320 à 1280 pixels) pour montrer :

- la vitesse (FPS) obtenue à chaque résolution,
- le nombre moyen de détections trouvées (un indicateur approximatif, pas une vraie
  mesure de précision scientifique).

Ça aide à choisir une résolution raisonnable avant de lancer l'application sur un
vrai flux de caméra.

## Onglet "About / Deployment"

Résume simplement les deux modes (local/distant) et renvoie vers le `README.md` du
dossier pour les instructions complètes d'hébergement (Hugging Face Spaces, VPS...).

## En résumé

| Question | Réponse |
|---|---|
| Ça sert à quoi ? | Tester en direct le modèle de détection EPI sur une caméra ou une vidéo |
| Où tourne le modèle ? | Au choix : sur cette machine, ou sur un serveur distant |
| Comment réduire les fausses alertes ? | Monter le seuil de confiance, utiliser le filtre temporel et les zones EPI |
| Comment savoir quelle résolution choisir ? | Utiliser l'onglet Benchmark sur sa propre vidéo |
