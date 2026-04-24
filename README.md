# gulf-stream-sonification
Sonifying Gulf Stream Surface Currents 2026

## Summary

This project aims to explore data sonification of ocean currents and results in a synth modulation mapping based on the EOF breakdown of the Gulf Stream currents from Jan to April of 2026, mapped onto an Alchemy instrument in Logic and bounced as an MP3

## Data and Exploration

Gulf Stream surface current velocity data was sourced from the Copernicus Marine Service (CMEMS) at hourly resolution across a Gulf Stream region from January to April 2026, providing U and V velocity components across a spatial grid of latitude and longitude points. Initial exploration included computing basin-mean current speed, Okubo-Weiss parameter fields to identify eddy-dominated versus jet-dominated regions, and a spectrogram of basin-mean speed to characterize the dominant periodicities — including M2 tidal (12.4h), diurnal K1 (24h), inertial (~28h at 25°N), and mesoscale variability at 60-140 hour periods. EOF decomposition was then applied to the spatiotemporal speed field to extract the dominant modes of variability and their temporal evolution.

## Sonification

The first four principal components from the EOF decomposition were mapped onto a six-voice D-centered chord in Logic Pro's Alchemy synthesizer, with bed notes (D4, E4, A4) providing a stable harmonic foundation and shimmer notes (B4, D5, F#5) driven by PC2, PC3, and PC4 respectively. Global modulations applied across all voices include PC1 controlling filter cutoff as the primary breathing motion, basin-mean speed controlling overall volume, and eddy fraction from the Okubo-Weiss parameter controlling reverb send to create spatial bloom during active eddy periods. A slow low-pass filtered version of PC1 drives pitch bend across all voices, creating a subtle alive drift that keeps the cluster from feeling static.

## Takeaways
Through this project, I developed working knowledge of ocean current dynamics, fluid dynamics modeling, and the signal processing methods used to characterize spatiotemporal variability in physical systems. 

Data sonification serves as a bridge between my interests of data and audio, and I learned how to map data to channels in Logic Pro, utilizing my taste in music to create a meaningful and intentional representation of the data, given the immense choice that exists in presenting data.

I hope to expand my modeling efforts for larger research projects that aim to solve problems and/or create predictive models.
