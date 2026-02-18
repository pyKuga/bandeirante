SELECT *
FROM cotacoes
WHERE CODNEG ~* '%[A-Z]{1}[0-9]{2,3}E*)$';


SELECT *
from cotacoes
where CODNEG = ?;

