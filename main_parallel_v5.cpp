// --- CONTENUTO PER main_parallel_v5.cpp ---

#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <omp.h>
#include "kmeans.h" // Il nostro header condiviso

int main() {
    // --- PARAMETRI DEL PROBLEMA ---
    const int NUM_POINTS = 5000000;
    const int K = 10;
    const int MAX_ITERATIONS = 30;

    // NUOVO (v5): Aggiungiamo un padding. 8 double = 64 byte (1 cache line)
    // Aggiungiamo 8 "double" vuoti alla fine di ogni array privato.
    const int PADDING = 8;

    // NUOVO: Messaggio aggiornato
    std::cout << "K-Means Parallelo (v5 - Padding / False Sharing Fix) Iniziato..." << std::endl;
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

    // --- Setup per la riduzione manuale (MODIFICATO) ---
    int max_threads = omp_get_max_threads();

    // NUOVO: Creiamo gli array privati con K + PADDING
    std::vector<std::vector<double>> private_sumX(max_threads, std::vector<double>(K + PADDING, 0.0));
    std::vector<std::vector<double>> private_sumY(max_threads, std::vector<double>(K + PADDING, 0.0));
    std::vector<std::vector<int>> private_count(max_threads, std::vector<int>(K + PADDING, 0));

    // NUOVO (v5): Creiamo gli array globali QUI, fuori dal timer
    std::vector<double> sumX(K, 0.0);
    std::vector<double> sumY(K, 0.0);
    std::vector<int> count(K, 0);


    // Inizia il timer
    double startTime = omp_get_wtime();

    // --- 4. ALGORITMO K-MEANS ---
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            // --- 1. RESET ---
            // Ogni thread resetta i SUOI array privati (inclusi i byte di padding)
            std::fill(private_sumX[tid].begin(), private_sumX[tid].end(), 0.0);
            std::fill(private_sumY[tid].begin(), private_sumY[tid].end(), 0.0);
            std::fill(private_count[tid].begin(), private_count[tid].end(), 0);

            #pragma omp barrier

            // --- 2. ASSEGNAZIONE (Passo A) ---
            #pragma omp for
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

            #pragma omp barrier

            // --- 3. ACCUMULAZIONE (Passo B.2) ---
            // Il codice qui è IDENTICO, ma grazie al padding
            // non c'è più false sharing.
            #pragma omp for
            for (int i = 0; i < NUM_POINTS; ++i) {
                int cluster = points[i].clusterId;
                private_sumX[tid][cluster] += points[i].x;
                private_sumY[tid][cluster] += points[i].y;
                private_count[tid][cluster]++;
            }

        } // --- Fine blocco 'parallel' ---


        // --- 4. RIDUZIONE E MEDIA (Passo B.3 & B.4) ---

        // NUOVO (v5): Non creiamo i vettori, li resettiamo.
        // Questo è più veloce (evita allocazione di memoria).
        std::fill(sumX.begin(), sumX.end(), 0.0);
        std::fill(sumY.begin(), sumY.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);

        // Somma i risultati privati
        for (int t = 0; t < max_threads; ++t) {
            for (int k = 0; k < K; ++k) { // Il ciclo 'k' si ferma a K (ignora il padding)
                sumX[k] += private_sumX[t][k];
                sumY[k] += private_sumY[t][k];
                count[k] += private_count[t][k];
            }
        }

        // Calcola la media
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

    std::cout << "--- K-Means Parallelo (v5) Terminato ---" << std::endl;
    std::cout << "Tempo di esecuzione parallelo (v5): " << parallelTime << " secondi" << std::endl;

    double sequentialTime = 48.1065;
    std::cout << "Speedup vs Sequenziale: " << sequentialTime / parallelTime << "x" << std::endl;

    std::cout << "Centroidi finali:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Cluster " << k << ": (" << centroids[k].x << ", " << centroids[k].y << ")" << std::endl;
    }

    return 0;
}