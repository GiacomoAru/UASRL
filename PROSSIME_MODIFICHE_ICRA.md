# Prossime modifiche verso ICRA

## Priorità di implementazione

1. Completare e validare la baseline ATOM-CBF.
2. Implementare e validare una versione locale di MPPI che usi soltanto le osservazioni disponibili alla policy.
3. **Alta priorità subito dopo MPPI:** verificare la generalità del metodo rispetto allo shield, applicando lo stesso uncertainty gate a correttori d'azione semplici e indipendenti dalla CBF.

## Test prioritario sulla generalità rispetto allo shield

Il primo correttore sarà un braking shield minimale. Quando viene attivato, ridurrà la velocità lineare proposta dalla policy secondo `v_safe = beta * v_policy`, mantenendo la velocità angolare entro i limiti del LIMO. Una seconda variante applicherà limiti operativi conservativi alla velocità lineare e angolare. I limiti fisici degli attuatori resteranno sempre attivi, mentre il gate controllerà soltanto le restrizioni conservative aggiuntive.

Il secondo correttore sarà un constant-action lookahead shield. Il filtro propagherà l'azione corrente `(v, omega)` per un breve orizzonte di `H` step con il modello cinematico unicycle e usando esclusivamente gli ostacoli osservati dai sensori locali. Quando la traiettoria prevista entrerà nel margine di sicurezza, il filtro ridurrà progressivamente la velocità lineare.

Per ogni correttore verranno confrontate una variante always-on e una uncertainty-gated con soglia `q = 0.90`. Policy, seed e ambienti resteranno identici. I parametri `beta`, `H`, passo di integrazione e margine di sicurezza saranno selezionati soltanto sugli ambienti ID e congelati prima dei test OOD.

Il confronto includerà policy nominale, braking shield, lookahead shield e CBF, ciascuno nelle configurazioni pertinenti. Le metriche principali saranno successo, collisioni, timeout, SEL, frequenza degli interventi e distanza tra azione nominale e azione eseguita. Un buon compromesso tra sicurezza ed efficienza con correttori diversi fornirebbe evidenza sperimentale diretta della generalità del metodo rispetto allo shield utilizzato.

## Idee successive

1. Eseguire brevi rollout con un world model per ottenere una stima di rischio o incertezza anticipata.
2. Studiare un QP che scelga la correzione dell'azione minimizzando l'incertezza prevista, invece di usare soltanto la distanza dagli ostacoli.
