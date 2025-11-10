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
    std::cout << "K-Means Parallelo (v2 - Atomic) Iniziato..." << std::endl;
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

        // --- B. PASSO DI AGGIORNAMENTO (ORA PARALLELO) ---
        std::vector<double> sumX(K, 0.0);
        std::vector<double> sumY(K, 0.0);
        std::vector<int> count(K, 0);

        // NUOVO: Parallelizziamo anche questo ciclo!
        #pragma omp parallel for
        for (int i = 0; i < NUM_POINTS; ++i) {
            int cluster = points[i].clusterId;

            // NUOVO: Usiamo 'atomic' per evitare race conditions
            #pragma omp atomic
            sumX[cluster] += points[i].x;

            #pragma omp atomic
            sumY[cluster] += points[i].y;

            #pragma omp atomic
            count[cluster]++;
        }

        // Questo piccolo ciclo (solo 10 iterazioni) è velocissimo,
        // lo lasciamo sequenziale. Parallelizzarlo sarebbe inutile.
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

    // NUOVO: Messaggi aggiornati
    std::cout << "--- K-Means Parallelo (v2) Terminato ---" << std::endl;
    std::cout << "Tempo di esecuzione parallelo (v2): " << parallelTime << " secondi" << std::endl;

    // Usiamo il nostro benchmark corretto
    double sequentialTime = 48.1065;
    std::cout << "Speedup vs Sequenziale: " << sequentialTime / parallelTime << "x" << std::endl;

    // Stampa i centroidi finali per un controllo di correttezza
    std::cout << "Centroidi finali:" << std::endl;
    for (int k = 0; k < K; ++k) {
        std::cout << "Cluster " << k << ": (" << centroids[k].x << ", " << centroids[k].y << ")" << std::endl;
    }

    return 0;
}