DECLARE @example NVARCHAR(30) = 'A.B.C.D'

SELECT value  
FROM STRING_SPLIT(@example, '.')  
WHERE RTRIM(value) <> '';