#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include "kmeans.h"
#include <omp.h> // NUOVO - Includiamo l'header di OpenMP per il timer



int main() {
    // --- PARAMETRI DEL PROBLEMA ---
    // Aumentiamo i punti per avere un tempo di esecuzione misurabile
    const int NUM_POINTS = 5000000; // NUOVO: 5 Milioni di punti
    const int K = 10;               // NUOVO: 10 cluster
    const int MAX_ITERATIONS = 30;

    std::cout << "K-Means Sequenziale Iniziato..." << std::endl;
    std::cout << "Punti: " << NUM_POINTS << ", Cluster: " << K << std::endl;

    // --- 2. GENERAZIONE DATI ---
    std::vector<Point> points(NUM_POINTS);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 100.0);

    for (int i = 0; i < NUM_POINTS; ++i) {
        points[i].x = dis(gen);
        points[i].y = dis(gen);
    }
    std::cout << "Dati generati." << std::endl;

    // --- 3. INIZIALIZZAZIONE CENTROIDI ---
    std::vector<Point> centroids(K);
    for (int i = 0; i < K; ++i) {
        centroids[i] = points[i];
    }
    std::cout << "Centroidi inizializzati." << std::endl;


    // NUOVO: Inizia il timer
    double startTime = omp_get_wtime();

    // --- 4. ALGORITMO K-MEANS ---
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        // --- A. PASSO DI ASSEGNAZIONE ---
        for (int i = 0; i < NUM_POINTS; ++i) {
            double minDistance = std::numeric_limits<double>::max();
            int bestCluster = -1;
            for (int k = 0; k < K; ++k) {
                // CORREZIONE BUG: Assicurati che la funzione distanceSquared sia corretta
                double dist = distanceSquared(points[i], centroids[k]);
                if (dist < minDistance) {
                    minDistance = dist;
                    bestCluster = k;
                }
            }
            points[i].clusterId = bestCluster;
        }

        // --- B. PASSO DI AGGIORNAMENTO ---
        std::vector<double> sumX(K, 0.0);
        std::vector<double> sumY(K, 0.0);
        std::vector<int> count(K, 0);

        for (int i = 0; i < NUM_POINTS; ++i) {
            int cluster = points[i].clusterId;
            sumX[cluster] += points[i].x;
            sumY[cluster] += points[i].y;
            count[cluster]++;
        }

        for (int k = 0; k < K; ++k) {
            if (count[k] > 0) {
                centroids[k].x = sumX[k] / count[k];
                centroids[k].y = sumY[k] / count[k];
            }
        }

        // Togliamo la stampa per non "sporcare" la misurazione del tempo
        // std::cout << "Iterazione " << iter + 1 << "/" << MAX_ITERATIONS << " completata." << std::endl;
    }

    // NUOVO: Ferma il timer
    double endTime = omp_get_wtime();
    double sequentialTime = endTime - startTime; // Calcola il tempo trascorso

    std::cout << "--- K-Means Sequenziale Terminato ---" << std::endl;

    // NUOVO: Stampa il tempo
    std::cout << "Tempo di esecuzione sequenziale: " << sequentialTime << " secondi" << std::endl;

    // (Opzionale) Stampa i centroidi finali
    std::cout << "Centroidi finali:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Cluster " << k << ": (" << centroids[k].x << ", " << centroids[k].y << ")" << std::endl;
    }

    return 0;
}