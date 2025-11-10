// --- CONTENUTO PER main_parallel_v4.cpp ---

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

    // NUOVO: Messaggio aggiornato
    std::cout << "K-Means Parallelo (v4 - Blocco Unico) Iniziato..." << std::endl;
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

    // --- Setup per la riduzione manuale (identico a v3) ---
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<double>> private_sumX(max_threads, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> private_sumY(max_threads, std::vector<double>(K, 0.0));
    std::vector<std::vector<int>> private_count(max_threads, std::vector<int>(K, 0));

    // Inizia il timer
    double startTime = omp_get_wtime();

    // --- 4. ALGORITMO K-MEANS ---
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        // --- NUOVO: "Raduna la squadra" UNA SOLA VOLTA ---
        #pragma omp parallel
        {
            // Ogni thread scopre il proprio ID
            int tid = omp_get_thread_num();

            // --- 1. RESET (ex Passo B.1) ---
            // Ogni thread resetta SOLO I SUOI array privati
            std::fill(private_sumX[tid].begin(), private_sumX[tid].end(), 0.0);
            std::fill(private_sumY[tid].begin(), private_sumY[tid].end(), 0.0);
            std::fill(private_count[tid].begin(), private_count[tid].end(), 0);

            // Sincronizzazione: aspetta che TUTTI i thread
            // abbiano finito di resettare prima di procedere.
            #pragma omp barrier

            // --- 2. ASSEGNAZIONE (Passo A) ---
            // 'for' (senza 'parallel') divide il lavoro tra
            // i thread del team GIA' ESISTENTE.
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

            // Sincronizzazione: aspetta che TUTTI i thread
            // abbiano finito l'Assegnazione prima di iniziare
            // l'Accumulazione (che dipende dai nuovi clusterId).
            #pragma omp barrier

            // --- 3. ACCUMULAZIONE (Passo B.2) ---
            #pragma omp for
            for (int i = 0; i < NUM_POINTS; ++i) {
                int cluster = points[i].clusterId;
                private_sumX[tid][cluster] += points[i].x;
                private_sumY[tid][cluster] += points[i].y;
                private_count[tid][cluster]++;
            }

            // La fine del blocco 'parallel' agisce come una
            // barriera implicita finale.

        } // --- "Sciogli la squadra" ---


        // --- 4. RIDUZIONE E MEDIA (Passo B.3 & B.4) ---
        // Questo è fatto dal thread principale (master),
        // è sequenziale e velocissimo.

        // Reset degli array globali
        std::vector<double> sumX(K, 0.0);
        std::vector<double> sumY(K, 0.0);
        std::vector<int> count(K, 0);

        // Somma i risultati privati
        for (int t = 0; t < max_threads; ++t) {
            for (int k = 0; k < K; ++k) {
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

    std::cout << "--- K-Means Parallelo (v4) Terminato ---" << std::endl;
    std::cout << "Tempo di esecuzione parallelo (v4): " << parallelTime << " secondi" << std::endl;

    double sequentialTime = 48.1065;
    std::cout << "Speedup vs Sequenziale: " << sequentialTime / parallelTime << "x" << std::endl;

    std::cout << "Centroidi finali:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Cluster " << k << ": (" << centroids[k].x << ", " << centroids[k].y << ")" << std::endl;
    }

    return 0;
}