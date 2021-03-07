# Mind

For the English version check: [README_ENGLISH.md](README_ENGLISH.md)

## Présentation

Le projet Mind est un projet réalisé dans le cadre de l'association PoC. Il a pour but de détecter et d'analyser les signaux mentaux ou "Brainwaves" grâce à l'aide d'un casque EEG (Electroencephalographe).

[![alt text](http://img.youtube.com/vi/12wdhqHyKLE/hqdefault.jpg)](https://youtu.be/12wdhqHyKLE)

La version actuelle du projet utilise le casque EEG opensource d'[OpenBCI](https://openbci.com/) : le [Ultracortex mark IV](https://shop.openbci.com/collections/frontpage/products/ultracortex-mark-iv). Le "Chipset" que l'on utilise est le [Cyton](https://docs.openbci.com/docs/02Cyton/CytonLanding) avec le [WifiShield](https://docs.openbci.com/docs/05ThirdParty/03-WiFiShield/WiFiLanding). Des membres de PoC ont imprimé et installé le casque dans une itération précédente.

## Fonctionnement

Le casque détecte les signaux et les diffuse sur son propre réseau wifi. L'ordinateur peut, en étant connecté à ce réseau, capter les données grâce à la GUI d'OpenBCI et les transmettre vers un flux LSL. Ces données peuvent ensuite être récupérées dans un script Python grâce à la librairie `pylsl`.

A partir de ces données nous avons créé deux datasets : `go` et `none`. description de l'IA

L'itération actuelle du projet utilise ces données en tandem avec une scene Unity à travers des sockets. 

expliquer FFT networking
expliquer widget

## Installation et utilisation

Une documentation complète des produit d'OpenBCI est [ici](https://docs.openbci.com/docs/Welcome.html).

Le projet à été testé et fonctionne sur Windows. Unity est instable sous les distributions Linux mais le projet devrait aussi fonctionner dans cet environnement.

#### Installation :
- Du casque :
  - Suivre le [tutoriel](https://docs.openbci.com/docs/04AddOns/01-Headwear/MarkIV) d'OpenBCI
- De la GUI :
  - suivre le [tutoriel](https://docs.openbci.com/docs/06Software/01-OpenBCISoftware/GUIDocs) d'OpenBCI
- De Unity :
  - Le téléchargement de [unity](https://store.unity.com/#plans-individual)
  - Comment utiliser unity : [documentation](https://docs.unity3d.com/Manual/index.html) et [vidéos](https://www.youtube.com/results?search_query=learn+unity+playlist)

#### Utilisation :
- Le casque (vue de dos):
  - Le bouton situé sur la droite du Cyton chipset doit être positionné sur `PC`
![PC]()
  - Le bouton situé en bas à gauche du WifiShield doit être positionné sur `ON`
![Ext-Pwr]()
  - Allumez ou éteignez le casque grâce aux valeurs `ON` et `OFF` du bouton situé à droite du WifiShield
![Power]()
  - Une fois allumé, le casque émet un réseau wifi auquel vous pourrez vous connecter avec votre ordinateur.
  - Positionnez le casque sur votre tête, chipset à l'arrière, et branchez les deux pinces sur vos lobes d'oreille
- La GUI :
  - Lancez la GUI
  - Dans le `System Control Panel` sélectionnez `CYTON (live)` puis `Wifi (from Wifi Shield)` et enfin `STATIC IP`
  - Lancez la session en appuyant sur `START SESSION`
  - Commencez à recevoir des données en sélectionnant le bouton vert : `START DATA STREAM`
- Le flux de données LSL :
  - Une fois votre flux principal de données lancé, changez votre widget `Accelerometer` en un widget `Networking` en ouvrant le menu déroulant qu'est nom du widget
  - Sélectionnez le menu déroulant `Serial` et choisissez `LSL`
  - Sous la rubrique `Stream 1`, clickez sur le menut déroulant `None` et sélectionnez `FFT`
  - Clickez sur `Start` pour lancer le flux de données LSL

#### Unity :
- Lancez le Unity Hub
- Ajoutez ce projet à vos projets Unity : clickez sur `ADD` et naviguez vers le dossier [unity](unity/)
- Lancez le projet avec une version compatible de Unity. Ce projet à été réalisé avec la version `2019.4.12f1`

Ca y est, votre donnée est récupérable facilement avec un script python et est déjà partiellement traitée (Donnée FFT). Les étapes suivantes ne sont applicable uniquement pour les scripts et fichiers de ce Dépot Github.

Les scripts et leur utilisation :
- [create_dataset.py](data/create_dataset.py) : créé un des datasets numpy de 1 seconde de la forme [25,8,60]
  - 25 : quantité moyenne de datapoints par seconde
  - 8 : nombre d'électrodes du casque
  - 60 : La donnée FFT divise la donnée sur 125 fréquences mais seules celles inférieures à 60 sont utilisable à cause du bruit électromagnétique 
- [mindTrain.py](mindTrain.py) : créé un modèle de machine learning à partir des fichier numpy
- [mindPred.py](mindPred.py) : lance des prédictions sur un set de données
- [real_time.py](real_time.py) : permet de la prédiction en temps réel
  - Une fois toutes les étapes de [l'installation](#Installation) et de [l'utilisation](#Utilisation) terminées, vous pouvez utiliser ce script pour prédire `go` ou `none` en temps réel
  - ```$> python3 real_time.py chemin_du_modèle.pt```
- [mindIM.py](mindIM.py) : prédiction en temps réel + utilisation avec unity
  - Une fois toutes les étapes de [l'installation](#Installation), de [l'utilisation](#Utilisation) et de [Unity](#Unity) terminées, vous pouvez utiliser ce script pour visualiser vos prédictions sur unity en temps réel
  - ```$> python3 mindIM.py```
  - Lancez la scène Unity
![start-unity-scene]()
