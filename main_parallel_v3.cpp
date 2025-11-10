// --- CONTENUTO PER main_parallel_v3.cpp ---

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
    std::cout << "K-Means Parallelo (v3 - Riduzione Manuale) Iniziato..." << std::endl;
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


    // --- NUOVO (v3): Setup per la riduzione manuale ---
    // Dobbiamo sapere quanti thread useremo al massimo
    int max_threads = omp_get_max_threads();

    // Creiamo gli array "privati" per ogni thread
    // Un vector di "K" zeri per ogni thread
    std::vector<std::vector<double>> private_sumX(max_threads, std::vector<double>(K, 0.0));
    std::vector<std::vector<double>> private_sumY(max_threads, std::vector<double>(K, 0.0));
    std::vector<std::vector<int>> private_count(max_threads, std::vector<int>(K, 0));


    // Inizia il timer
    double startTime = omp_get_wtime();

    // --- 4. ALGORITMO K-MEANS ---
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        // --- A. PASSO DI ASSEGNAZIONE (PARALLELO) ---
        // Questo è già ottimizzato
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

        // --- B. PASSO DI AGGIORNAMENTO (PARALLELO v3) ---

        // 1. Reset degli array privati per ogni iterazione
        // (dato che #pragma omp parallel for può riusare i thread)
        // Dobbiamo resettare solo la porzione che verrà usata
        #pragma omp parallel for
        for(int t=0; t < max_threads; ++t) {
            std::fill(private_sumX[t].begin(), private_sumX[t].end(), 0.0);
            std::fill(private_sumY[t].begin(), private_sumY[t].end(), 0.0);
            std::fill(private_count[t].begin(), private_count[t].end(), 0);
        }

        // 2. Calcolo parallelo su array privati (NO ATOMIC)
        #pragma omp parallel for
        for (int i = 0; i < NUM_POINTS; ++i) {
            int tid = omp_get_thread_num(); // Chiediamo: "Quale thread sono?"
            int cluster = points[i].clusterId;

            // Aggiorniamo l'array privato del NOSTRO thread
            // NESSUNA race condition, NESSUN atomic!
            private_sumX[tid][cluster] += points[i].x;
            private_sumY[tid][cluster] += points[i].y;
            private_count[tid][cluster]++;
        }

        // 3. Riduzione finale (sequenziale, ma velocissima)
        // Creiamo gli array "globali"
        std::vector<double> sumX(K, 0.0);
        std::vector<double> sumY(K, 0.0);
        std::vector<int> count(K, 0);

        // Per ogni thread...
        for (int t = 0; t < max_threads; ++t) {
            // ...e per ogni cluster, aggiungi i risultati privati al totale
            for (int k = 0; k < K; ++k) {
                sumX[k] += private_sumX[t][k];
                sumY[k] += private_sumY[t][k];
                count[k] += private_count[t][k];
            }
        }

        // 4. Calcolo media (sequenziale, velocissimo)
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

    std::cout << "--- K-Means Parallelo (v3) Terminato ---" << std::endl;
    std::cout << "Tempo di esecuzione parallelo (v3): " << parallelTime << " secondi" << std::endl;

    double sequentialTime = 48.1065;
    std::cout << "Speedup vs Sequenziale: " << sequentialTime / parallelTime << "x" << std::endl;

    std::cout << "Centroidi finali:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Cluster " << k << ": (" << centroids[k].x << ", " << centroids[k].y << ")" << std::endl;
    }

    return 0;
}