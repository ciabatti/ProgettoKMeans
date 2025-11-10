// --- CONTENUTO PER main_parallel_v1.cpp ---

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include "kmeans.h"
#include <omp.h> // Header OpenMP



int main() {
    // --- PARAMETRI DEL PROBLEMA ---
    const int NUM_POINTS = 5000000;
    const int K = 10;
    const int MAX_ITERATIONS = 30;

    // Messaggio cambiato
    std::cout << "K-Means Parallelo (v1) Iniziato..." << std::endl;
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

    // Inizia il timer
    double startTime = omp_get_wtime();

    // --- 4. ALGORITMO K-MEANS ---
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        // --- A. PASSO DI ASSEGNAZIONE (PARALLELO) ---

        // ! ! ! MODIFICA CHIAVE ! ! !
        #pragma omp parallel for
        for (int i = 0; i < NUM_POINTS; ++i) {
            double minDistance = std::numeric_limits<double>::max();
            int bestCluster = -1;

            for (int k = 0; k < K; ++k) {
                double dist = distanceSquared(points[i], centroids[k]);
                if (dist < minDistance) {
                    minDistance = dist;
                    bestCluster = k;
                }
            }
            points[i].clusterId = bestCluster;
        }

        // --- B. PASSO DI AGGIORNAMENTO (Ancora sequenziale) ---
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
    }

    // Ferma il timer
    double endTime = omp_get_wtime();
    double parallelTime = endTime - startTime;

    std::cout << "--- K-Means Parallelo (v1) Terminato ---" << std::endl;

    // Stampa aggiornata
    std::cout << "Tempo di esecuzione parallelo (v1): " << parallelTime << " secondi" << std::endl;

    // Calcoliamo lo speedup!
    double sequentialTime = 48.1687; // Il tuo benchmark!
    std::cout << "Speedup vs Sequenziale: " << sequentialTime / parallelTime << "x" << std::endl;

    // (Opzionale) Stampa i centroidi finali
    std::cout << "Centroidi finali:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Cluster " << k << ": (" << centroids[k].x << ", " << centroids[k].y << ")" << std::endl;
    }

    return 0;
}