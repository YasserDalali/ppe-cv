---
source_pdf: PPE_Systeme_FR.pdf
pages: 11
converted_with: pdftotext -layout
---

# PPE_Systeme_FR.pdf

> Auto-converted from PDF. Prefer this file over the PDF for AI reading.

---

        CONFORMITÉ EPI
        SYSTÈME DE VISION AUTOMATISÉ

Réduire le coût humain et économique de la non-conformité EPI —
au-delà de ce qu'un humain peut repérer en surveillant une caméra.




OCP • NOTE TECHNIQUE SÉCURITÉ


---

                  LES ENJEUX



LE PRIX DE
REGARDER AILLEURS

   2,78 M                                                                                                374 M
   Décès par an dus aux accidents et maladies professionnels                                             Blessures non mortelles au travail enregistrées chaque année,
   dans le monde                                                                                         dans le monde




   3 000 Mds $                                                                                           1 Md $+
   Coût mondial annuel des accidents et maladies au travail —                                            Coût des accidents du travail aux États-Unis, chaque semaine
   3,94 % du PIB mondial



Source : Organisation internationale du Travail, Sécurité et santé au travail (2023) ; estimations du National Safety Council (É.-U.).                                   02


---

   LE GOULOT D'ÉTRANGLEMENT



REGARDER UN ÉCRAN N'EST
PAS UN PLAN DE SÉCURITÉ

                                                                                            TAUX DE DÉTECTION SELON LE TEMPS
 L'attention soutenue s'effondre en 20 à 35 minutes de surveillance continue — c'est la
 baisse de vigilance (Parasuraman, 1984 ; Sawin & Scerbo, 1995).




 Après 20 minutes de visionnage actif, les opérateurs manquent jusqu'à 95 % de l'activité
 affichée à l'écran (Velastin et al., 2006).




 Des opérateurs CCTV formés, ﬁlmés dans de vraies usines de traitement de minerai,
 n'ont détecté qu'environ 50 % des comportements à risque ciblés — avec un taux élevé
 de fausses alertes (Donald et al., Applied Ergonomics, 2015).
                                                                                            Courbe illustrative construite à partir des ordres de grandeur rapportés dans les
                                                                                            études citées sur la baisse de vigilance.


 Sur une session de 90 minutes, les pertes d'attention touchent 23 % des opérateurs
 dans les 30 premières minutes, puis 60 % dans les 30 suivantes (Sandia National
 Laboratories, 2014).
                                                                                                                                                                                03


---

LE CALCUL



UNE MINUTE
DÉCIDE DE L'ANNÉE
 525 600                                            1                                                   ~20 min
 minutes de vidéo qu'un même surveillant            instant dans cette année où un accident             avant que la vigilance ne s'effondre — au-delà,
 regarde en une année                               survient réellement                                 l'image est regardée, pas vue




  Un accident est un instant précis. Une année de surveillance ne « paie » que si un humain se trouve, par hasard, pleinement attentif à cette
  minute-là. Une fois l'attention émoussée — ce qui, selon la recherche, survient en moins de 20 minutes — chaque heure précédente de
  visionnage ne rapporte aucune valeur récupérée. Une année de surveillance manuelle est un pari joué sur un seul tirage.




                                                                                                                                                      04


---

              LE SYSTÈME



COMMENT FONCTIONNE
NOTRE SYSTÈME
Quatre étapes entre un ﬂux caméra et une alerte conﬁrmée.




 01                                 02                              03                               04
 MODÈLE DE BASE                     DONNÉES OCP                     JUGE LLM                         ALERTE
 Modèle de détection EPI            Affiné sur nos propres images   Une passe de raisonnement sur    Violations conﬁrmées
 pré-entraîné — une base de         de site auto-annotées —         chaque détection signalée pour   transmises instantanément —
 détection d'objets éprouvée.       équipements et conditions       réduire les faux positifs.       sur les caméras déjà installées.
                                    réels.




                                                                                                                                        05


---

LE SYSTÈME




             05


---

          ÉTAPE 01 – 02



VOIR CE QUE LES MODÈLES
GÉNÉRIQUES RATENT

 MODÈLE STANDARD                                                    ADAPTÉ AVEC LES DONNÉES OCP
  •   Entraîné sur des jeux de données publics génériques            •   Jeu de données auto-annoté construit à partir de vraies
                                                                         images des caméras OCP
  •   N'a jamais vu les caméras, angles ou l'éclairage d'OCP, ni
      la poussière de phosphate                                      •   Affiné sur nos sites, équipements et conditions de
                                                                         poussière/lumière exactes
  •   Ne reconnaît pas les équipements et tenues propres au
      site                                                           •   Apprend à quoi ressemble une véritable violation sur
                                                                         notre terrain
  •   Résultat : faible rappel, faible précision, sur nos propres
      images                                                         •   Résultat : rappel et précision plus élevés là où il tourne
                                                                         réellement




                                                                                                                                      06


---

ÉTAPE 01 – 02


---



---

            ÉTAPE 03



UN JUGE LLM POUR
ÉLIMINER LES FAUSSES ALERTES


           LE DÉTECTEUR                                           LE JUGE LLM                                             VIOLATION
          SIGNALE UN CAS                                      ANALYSE LE CONTEXTE                                   CONFIRMÉE UNIQUEMENT




  •   Chaque détection brute est jugée avant de devenir une alerte : occlusion, reﬂets, distance et éclairage sont raisonnés, pas seulement
      notés.
  •   La vidéosurveillance manuelle génère déjà un taux élevé de fausses alertes (Donald et al., 2015) — un détecteur non ﬁltré aggraverait la
      fatigue liée aux alertes.
  •   Seules les violations conﬁrmées par le juge atteignent un humain — chaque alerte mérite d'être traitée.




                                                                                                                                                 07


---

ÉTAPE 04 — DÉPLOIEMENT



LE SYSTÈME NE
CLIGNE JAMAIS DES YEUX
 ALERTES EN TEMPS RÉEL                          MATÉRIEL EXISTANT                              AUCUNE LIMITE DE 20 MIN
 Violations conﬁrmées transmises                Fonctionne sur les caméras et le réseau déjà   Pas de baisse de vigilance, pas de fatigue
 instantanément aux responsables sécurité,      installés sur site. Aucune nouvelle caméra,    d'équipe, pas de minute manquée. Chaque
 avec un extrait horodaté — à l'instant même.   aucun nouvel investissement.                   caméra, chaque minute, toute l'année.




                            Moins de blessures. Moins de pertes. Aucune minute manquée.

                                                                                                                                            08


---

