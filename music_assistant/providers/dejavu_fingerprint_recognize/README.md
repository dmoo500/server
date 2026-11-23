# Dejavu Fingerprint Recognizer

This experimental audio-analysis provider identifies music from explicitly selected live inputs. It uses Music Assistant's existing live-source analysis and metadata reporting; it does not add tracks to the library or alter a queue. Only inputs that allow audio analysis can be recognized.

The provider keeps up to 15 seconds of PCM in memory and sends locally extracted landmarks, not audio, to the configured fingerprint server. It starts with three seconds and, after an inconclusive result, tries five, ten and fifteen seconds at five-second intervals. After an unsuccessful fifteen-second query it waits 30 seconds. After a match with valid timing, it checks near the expected end of the track or after at most 60 seconds of audio; a skip can therefore take up to 60 seconds to detect. Without valid timing it checks after 30 seconds. Requests across sources are paced and server retry instructions are respected.

Configure a fingerprint server URL and a read key during setup. An application key is also needed for the APISIX gateway. Select the live inputs in the provider's options after setup. HTTPS is required except for loopback HTTP. Recognition requires an existing matching reference on the server. Results include title, artist and album, and a position and duration when available. Music Assistant can look up artwork using its library and metadata providers when metadata lookup is enabled; artwork is not guaranteed.

## Manual Sendspin test

Use a separate test Music Assistant instance and data directory. Do not copy production credentials or modify an existing installation without permission. Configure its fingerprint server credentials and explicitly select the Sendspin input. Play the input through a player, using recordings already present in the fingerprint server, and check that the title, artist and album appear. Test a known song, a song absent from the reference database, a track change without silence, silence, and stopping and restarting the input. Verify the previous song is not shown as a new match and that playback continues without interruptions.

Test a temporary server outage and an invalid key separately: the provider logs the failure instead of reporting it as a match. A rate-limited server delays subsequent requests. A real Sendspin test is still needed to measure recognition latency and reliability; a synthetic local test does not establish accuracy on vinyl or real audio.
