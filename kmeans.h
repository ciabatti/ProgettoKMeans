#pragma once // Direttiva fondamentale: impedisce di includere questo file più volte

#include <cmath> // Ci serve per i calcoli, anche se per ora usiamo solo moltiplicazioni

/*
 * Questa è la nostra "cassetta degli attrezzi" condivisa
 * per l'algoritmo K-Means.
 */

// 1. La struttura dati per i punti
struct Point {
    double x = 0.0;
    double y = 0.0;
    int clusterId = -1; // A quale cluster appartiene?
};

// 2. La funzione per calcolare la distanza
// La dichiariamo 'inline' perché la sua definizione
// si trova in un file header.
inline double distanceSquared(Point p1, Point p2) {
    // La versione CORRETTA
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}