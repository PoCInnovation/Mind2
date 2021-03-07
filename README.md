# Mind

For the English version check: [README_ENGLISH.md](README_ENGLISH.md)

## Présentation

Le projet Mind est un projet réalisé dans le cadre de l'association PoC. Il a pour but de détecter et d'analyser les signaux mentaux ou "Brainwaves" grâce à l'aide d'un casque EEG (Electroencephalographe).

[![alt text](http://img.youtube.com/vi/12wdhqHyKLE/hqdefault.jpg)](https://youtu.be/12wdhqHyKLE)

La version actuelle du projet utilise le casque EEG opensource d'[OpenBCI](https://openbci.com/) : le [Ultracortex mark IV](https://shop.openbci.com/collections/frontpage/products/ultracortex-mark-iv). Des membres de PoC l'ont imprimé et installé dans une itération précédente.

## Fonctionnement

Le casque détecte les signaux et les diffuse sur son propre réseau wifi. L'ordinateur peut, en étant connecté à ce réseau, capter les données grâce à la GUI d'OpenBCI et les transmettre vers un flux LSL. Ces données peuvent ensuite être récupérées dans un script Python grâce à la librairie `pylsl`.

A partir de ces données nous avons créé deux datasets : `go` et `none`. description de l'IA

L'itération actuelle du projet utilise ces données en tandem avec une scene Unity à travers des sockets. 

## Installation et utilisation

- Du casque
  - Suivre le [tutoriel](https://docs.openbci.com/docs/04AddOns/01-Headwear/MarkIV) d'OpenBCI
- De la GUI