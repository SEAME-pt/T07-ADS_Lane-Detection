/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Zombie.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: joaoalme <joaoalme@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/08/22 15:01:44 by joaoalme          #+#    #+#             */
/*   Updated: 2023/08/22 15:01:45 by joaoalme         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */


#ifndef ZOMBIE_HPP
#	define ZOMBIE_HPP

#include <iostream>

class Zombie 
{
	private:
		std::string  name;
	
	public:
		Zombie(std::string z_name);
		~Zombie();
		void announce( void );

};
void    randomChump(std::string name);
Zombie *newZombie( std::string name );

#endif
