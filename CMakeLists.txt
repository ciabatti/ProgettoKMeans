cmake_minimum_required(VERSION 3.15)
project(ProgettoKMeans)

set(CMAKE_CXX_STANDARD 17)

# Trova e configura OpenMP
find_package(OpenMP REQUIRED)

# === QUESTA PARTE E' NUOVA E FONDAMENTALE ===
# Aggiunge esplicitamente i flag di OpenMP (es. -fopenmp)
# sia alla compilazione che al linking.
if(OpenMP_FOUND)
    message(STATUS "OpenMP Trovato! Imposto i flag CXX...")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
# ============================================

# Per eseguire la versione sequenziale, decommenta questa riga:
 #add_executable(ProgettoKMeans main.cpp)

# Per eseguire la versione parallela v1, decommenta questa riga:
#add_executable(ProgettoKMeans main_parallel_v1.cpp kmeans.h)

# Per eseguire la versione parallela v2, decommenta questa riga:
#add_executable(ProgettoKMeans main_parallel_v2.cpp kmeans.h)

# Per eseguire la versione parallela v3, decommenta questa riga: -- NON BUONO --
#add_executable(ProgettoKMeans main_parallel_v3.cpp kmeans.h)

# Per eseguire la versione parallela v4, decommenta questa riga:
add_executable(ProgettoKMeans main_parallel_v4.cpp kmeans.h)

# Per eseguire la versione parallela v5, decommenta questa riga:
#add_executable(ProgettoKMeans main_parallel_v5.cpp kmeans.h)


# Collega le librerie (ridondante ma sicuro)
target_link_libraries(ProgettoKMeans PRIVATE OpenMP::OpenMP_CXX)